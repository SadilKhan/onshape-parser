import pprint
import os
import copy
import numpy as np
from collections import OrderedDict
from utils import angle_from_vector_to_x, deg2rad, find_numeric, xyz_list2dict
from loguru import logger
from logger import OnshapeParserLogger
import json
from myclient import MyClient
from rich import print
# OnShape naming to Fusion360 naming format
EXTENT_TYPE_MAP = {'BLIND': 'OneSideFeatureExtentType', 'SYMMETRIC': 'SymmetricFeatureExtentType'}
OPERATION_MAP = {'NEW': 'NewBodyFeatureOperation', 'ADD': 'JoinFeatureOperation',
                 'REMOVE': 'CutFeatureOperation', 'INTERSECT': 'IntersectFeatureOperation'}


onshapeLogger=OnshapeParserLogger().configure_logger().logger

class FeatureListParser(object):
    """A parser for OnShape feature list (construction sequence)"""
    def __init__(self, client, did, wid, eid, data_id=None):
        self.c = client

        self.did = did
        self.wid = wid
        self.eid = eid
        self.data_id = data_id

        self.feature_list = self.c.get_features(did, wid, eid).json()

        with open("output/feature.json",'w') as f:
            json.dump(self.feature_list, f)

        self.profile2sketch = {}

    @staticmethod
    def parse_feature_param(feat_param_data):
        param_dict = {}
        for i, param_item in enumerate(feat_param_data):
            param_msg = param_item['message']
            param_id = param_msg['parameterId']
            if 'queries' in param_msg:
                param_value = []
                for i in range(len(param_msg['queries'])):
                    # Values look like  "JGG"
                    param_value.extend(param_msg['queries'][i]['message']['geometryIds']) # FIXME: could be error-prone
            # Values look like "10 in"
            elif 'expression' in param_msg:
                param_value = param_msg['expression']
            elif 'value' in param_msg:
                param_value = param_msg['value']
            else:
                raise NotImplementedError('param_msg:\n{}'.format(param_msg))

            param_dict.update({param_id: param_value})
        return param_dict
     
    @logger.catch()
    def _parse_sketch(self, feature_data):
        sket_parser = SketchParser(self.c, feature_data, self.did, self.wid, self.eid)
        save_dict = sket_parser.parse_to_fusion360_format()
        return save_dict

    def _expr2meter(self, expr):
        """convert value expresson to meter unit"""
        return self.c.expr2meter(self.did, self.wid, self.eid, expr)

    def _locateSketchProfile(self, geo_ids):
        return [{"profile": k, "sketch": self.profile2sketch[k]} for k in geo_ids]

    @logger.catch()
    def _parse_extrude(self, feature_data):
        param_dict = self.parse_feature_param(feature_data['parameters'])
        if 'hasOffset' in param_dict and param_dict['hasOffset'] is True:
            raise NotImplementedError("extrude with offset not supported: {}".format(param_dict['hasOffset']))

        entities = param_dict['entities'] # geometryIds for target face
        profiles = self._locateSketchProfile(entities)

        extent_one = self._expr2meter(param_dict['depth'])
        if param_dict['endBound'] == 'SYMMETRIC':
            extent_one = extent_one / 2
        if 'oppositeDirection' in param_dict and param_dict['oppositeDirection'] is True:
            extent_one = -extent_one
        extent_two = 0.0
        if param_dict['endBound'] not in ['BLIND', 'SYMMETRIC']:
            raise NotImplementedError("endBound type not supported: {}".format(param_dict['endBound']))
        elif 'hasSecondDirection' in param_dict and param_dict['hasSecondDirection'] is True:
            if param_dict['secondDirectionBound'] != 'BLIND':
                raise NotImplementedError("secondDirectionBound type not supported: {}".format(param_dict['endBound']))
            extent_type = 'TwoSidesFeatureExtentType'
            extent_two = self._expr2meter(param_dict['secondDirectionDepth'])
            if 'secondDirectionOppositeDirection' in param_dict \
                and str(param_dict['secondDirectionOppositeDirection']) == 'true':
                extent_two = -extent_two
        else:
            extent_type = EXTENT_TYPE_MAP[param_dict['endBound']]

        operation = OPERATION_MAP[param_dict['operationType']]

        save_dict = {"name": feature_data['name'],
                    "type": "ExtrudeFeature",
                    "profiles": profiles,
                    "operation": operation,
                    "start_extent": {"type": "ProfilePlaneStartDefinition"},
                    "extent_type": extent_type,
                    "extent_one": {
                        "distance": {
                            "type": "ModelParameter",
                            "value": extent_one,
                            "name": "none",
                            "role": "AlongDistance"
                        },
                        "taper_angle": {
                            "type": "ModelParameter",
                            "value": 0.0,
                            "name": "none",
                            "role": "TaperAngle"
                        },
                        "type": "DistanceExtentDefinition"
                    },
                    "extent_two": {
                        "distance": {
                            "type": "ModelParameter",
                            "value": extent_two,
                            "name": "none",
                            "role": "AgainstDistance"
                        },
                        "taper_angle": {
                            "type": "ModelParameter",
                            "value": 0.0,
                            "name": "none",
                            "role": "Side2TaperAngle"
                        },
                        "type": "DistanceExtentDefinition"
                    },
                    }
        return save_dict

    def _parse_boundingBox(self):
        bbox_info = self.c.eval_boundingBox(self.did, self.wid, self.eid)
        result = {"type": "BoundingBox3D",
                  "max_point": xyz_list2dict(bbox_info['maxCorner']),
                  "min_point": xyz_list2dict(bbox_info['minCorner'])}
        return result
    

    def _locateAxis(self,geo_id):
        """
        Parsing Axis information for revolution

        """
        axis_dict={}
        response=self.c.get_entity_by_id(self.did,self.wid,self.eid,geo_id,"EDGE").json()

        message=response['result']['message']['value']

        axis_dict = {}
        for item in message:
            for entry in item["message"]["value"]:
                key = entry['message']["key"]["message"]["value"]
                if key in ['direction','origin']:
                    value_list = entry['message']["value"]["message"]["value"]
                    val=[]
                    for vl in value_list:
                        val.append(vl['message']['value'])

                axis_dict[key] = val

        return axis_dict
    
    @logger.catch()
    def parse(self):
        """parse into fusion360 gallery format, 
        only sketch and extrusion are supported.
        """
        result = {"entities": OrderedDict(), "properties": {}, "sequence": []}

        # Bounding Box Parsing
        try:
            bbox = self._parse_boundingBox()
        except Exception as e:
            print(self.data_id, "bounding box failed:", e)
            return result
        result["properties"].update({"bounding_box": bbox})

        for i, feat_item in enumerate(self.feature_list['features']):
            feat_data = feat_item['message']
            # Type of Features(newSketch, revolve, extrude)
            feat_type = feat_data['featureType']
            feat_Id = feat_data['featureId']

            try:
                if feat_type == 'newSketch':
                    feat_dict = self._parse_sketch(feat_data)
                    for k in feat_dict['profiles'].keys():
                        self.profile2sketch.update({k: feat_Id})
                elif feat_type == 'extrude':
                    feat_dict = self._parse_extrude(feat_data)
                elif feat_type == 'revolve':
                    feat_dict=self._parse_revolve(feat_data)
                elif feat_type == 'fillet':
                    feat_dict=self._parse_fillet(feat_data)
                elif feat_type == "chamfer":
                    feat_dict=self._parse_chamfer(feat_data)
                else:
                    raise NotImplementedError(self.data_id, "unsupported feature type: {}".format(feat_type))
            except Exception as e:
                onshapeLogger.error(f"parse feature failed: {self.data_id} with error {e}")
                break
            result["entities"].update({feat_Id: feat_dict})
            result["sequence"].append({"index": i, "type": feat_dict['type'], "entity": feat_Id})
        return result

    @logger.catch()
    def _parse_revolve(self,feature_data):
        """
        {'bodyType': 'SOLID',
        'operationType': 'REMOVE',
        'entities': ['JKC'],
        'surfaceEntities': [],
        'axis': ['JJF'],
        'revolveType': 'FULL',
        'oppositeDirection': True,
        'angle': '30.0*deg',
        'angleBack': '30.0*deg',
        'defaultScope': True,
        #'booleanScope': ['JHD'],
        'asVersion': 'V23_PROCEDURAL_SWEPT_SURFACES'}
        
        """
        


        # SADIL: WORK NEEDS TO BE DONE
        param_dict=self.parse_feature_param(feature_data['parameters'])
        operation = OPERATION_MAP[param_dict['operationType']]

        entities = param_dict['entities'] # geometryIds for target face
        profiles = self._locateSketchProfile(entities)
        axis_dict=self._locateAxis(param_dict['axis'])
        # FIXME: Not using booleanscope 
        #booleanScope=self._locateSketchProfile(param_dict['booleanScope'])            
        # with open("output/revolve.json","w") as f:
        #     json.dump(feature_data, f)

        save_dict={
            "name":feature_data['name'],
            "type":"RevolveFeature",
            "profiles": profiles,
            "origin":axis_dict['origin'],
            "axis":axis_dict['direction'],
            #"booleanScope":booleanScope,
            'surfaceEntities': param_dict['surfaceEntities'],
            "operation": operation,
            "revolveType":param_dict['revolveType'],
            "oppositeDirection":param_dict['oppositeDirection'],
            "angle":find_numeric(param_dict['angle']),
            "angleBack":find_numeric(param_dict['angleBack']),
            "defaultScope":param_dict['defaultScope']
        }
        return save_dict
    
    @logger.catch()
    def _parse_fillet(self,feature_data):
        """
        {'entities': ['JQK', 'JQG'],
        'radius': '0.05*in',
        'tangentPropagation': True,
        'rho': '0.5',
        'asVersion': 'V608_MERGE_FROM_TOOLS',
        'allowEdgeOverflow': False}
        """
        param_dict=self.parse_feature_param(feature_data['parameters'])
        profiles=self._locateSketchProfile(param_dict['entities'])
        radius=self._expr2meter(param_dict['radius'])

        save_dict={
            "name":feature_data['name'],
            "type":"FilletFeature",
            "profiles":profiles,
            "radius":radius,
            "rho":0.5,
            'tangentPropagation': param_dict['tangentPropagation'],
            "allowEdgeOverflow": param_dict['allowEdgeOverflow']
        }
        return save_dict
    
    def _parse_chamfer(self):
        # SADIL: WORK NEEDS TO BE DONE
        pass

    def _parse_loft(self):
        raise NotImplementedError("Not Supported yet")

class SketchParser(object):
    """A parser for OnShape sketch feature list"""
    def __init__(self, client, feat_data, did, wid, eid, data_id=None):
        self.c = client
        self.feat_id = feat_data['featureId']
        self.feat_name = feat_data['name']
        self.feat_param = FeatureListParser.parse_feature_param(feat_data['parameters'])

        self.did = did
        self.wid = wid
        self.eid = eid
        self.data_id = data_id

        geo_id = self.feat_param["sketchPlane"][0]
        response = self.c.get_entity_by_id(did, wid, eid, [geo_id], "FACE")
        self.plane = self.c.parse_face_msg(response.json()['result']['message']['value'])[0]
        #self.plane={key.decode("utf-8"): value for key, value in self.plane.items() if type(key)!=str}


        self.geo_topo = self.c.eval_sketch_topology_by_adjacency(did, wid, eid, self.feat_id)
        #self.geo_topo={key.decode("utf-8"): value for key, value in self.geo_topo.items() if type(key)!=str}
        self._to_local_coordinates()
        self._build_lookup()

    def _to_local_coordinates(self):
        """transform into local coordinate system"""
        self.origin = np.array(self.plane["origin"])
        self.z_axis = np.array(self.plane["normal"])
        self.x_axis = np.array(self.plane["x"])
        self.y_axis = np.cross(self.plane["normal"], self.plane["x"])
        for item in self.geo_topo["vertices"]:
            old_vec = np.array(item["param"]["Vector"])
            new_vec = old_vec - self.origin
            item["param"]["Vector"] = [np.dot(new_vec, self.x_axis), 
                                       np.dot(new_vec, self.y_axis), 
                                       np.dot(new_vec, self.z_axis)]

        for item in self.geo_topo["edges"]:
            if item["param"]["type"] == "Circle":
                old_vec = np.array(item["param"]["coordSystem"]["origin"])
                new_vec = old_vec - self.origin
                item["param"]["coordSystem"]["origin"] = [np.dot(new_vec, self.x_axis),
                                                          np.dot(new_vec, self.y_axis),
                                                          np.dot(new_vec, self.z_axis)]

    def _build_lookup(self):
        """build a look up table with entity ID as key"""
        edge_table = {}
        for item in self.geo_topo["edges"]:
            edge_table.update({item["id"]: item})
        self.edge_table = edge_table

        vert_table = {}
        for item in self.geo_topo["vertices"]:
            vert_table.update({item["id"]: item})
        self.vert_table = vert_table

    def _parse_edges_to_loops(self, all_edge_ids):
        """sort all edges of a face into loops."""
        #onshapeLogger.info(f"All Edge Ids {all_edge_ids}")
        # FIXME: this can be error-prone. bug situation: one vertex connected to 3 edges
        vert2edge = {}
        if len(all_edge_ids)==1:
            return [[self.edge_table[all_edge_ids[0]]['id']]]
        for edge_id in all_edge_ids:
            item = self.edge_table[edge_id]
            if "vertices" in item: # Circle doesn't have vertices
                for vert in item["vertices"]: # Looks like ['JGE','JGY']
                    if vert not in vert2edge.keys():
                        vert2edge.update({vert: [item["id"]]})
                    else:
                        vert2edge[vert].append(item["id"])

        all_loops = []
        unvisited_edges = copy.copy(all_edge_ids)
        while len(unvisited_edges) > 0:
            cur_edge = unvisited_edges[0]
            unvisited_edges.remove(cur_edge)
            loop_edge_ids = [cur_edge]
            if "vertices" in self.edge_table[cur_edge] and len(self.edge_table[cur_edge]["vertices"]) == 0:  # no corresponding vertices
                pass
            #elif self.edge_table[cur_edge]['param']['type'].lower() == 'circle':
                #loop_edge_ids.append(cur_edge)
            elif "vertices" in self.edge_table[cur_edge]:
                loop_start_point, cur_end_point = self.edge_table[cur_edge]["vertices"][0], \
                                                  self.edge_table[cur_edge]["vertices"][-1]
                while cur_end_point != loop_start_point:
                    # find next connected edge
                    edges = vert2edge[cur_end_point][:]
                    edges.remove(cur_edge)
                    cur_edge = edges[0]
                    loop_edge_ids.append(cur_edge)
                    unvisited_edges.remove(cur_edge)

                    # find next enc_point
                    points = self.edge_table[cur_edge]["vertices"][:]
                    points.remove(cur_end_point)
                    cur_end_point = points[0]
            all_loops.append(loop_edge_ids)
        return all_loops

    def _parse_edge_to_fusion360_format(self, edge_id):
        """parse a edge into fusion360 gallery format. Only support 'Line', 'Circle' and 'Arc'."""
        edge_data = self.edge_table[edge_id]
        edge_type = edge_data["param"]["type"]
        #onshapeLogger.debug(f"Edge type: {edge_type}")
        if edge_type == "Line":
            start_id, end_id = edge_data["vertices"]
            start_point = xyz_list2dict(self.vert_table[start_id]["param"]["Vector"])
            end_point = xyz_list2dict(self.vert_table[end_id]["param"]["Vector"])
            curve_dict = OrderedDict({"type": "Line3D", "start_point": start_point,
                                      "end_point": end_point, "curve": edge_id})
        elif edge_type == "Circle" and "vertices" in edge_data == 2: # an Arc
            radius = edge_data["param"]["radius"]
            start_id, end_id = edge_data["vertices"]
            start_point = xyz_list2dict(self.vert_table[start_id]["param"]["Vector"])
            end_point = xyz_list2dict(self.vert_table[end_id]["param"]["Vector"])
            center_point = xyz_list2dict(edge_data["param"]["coordSystem"]["origin"])
            normal = xyz_list2dict(edge_data["param"]["coordSystem"]["zAxis"])

            start_vec = np.array(self.vert_table[start_id]["param"]["Vector"]) - \
                        np.array(edge_data["param"]["coordSystem"]["origin"])
            end_vec = np.array(self.vert_table[end_id]["param"]["Vector"]) - \
                      np.array(edge_data["param"]["coordSystem"]["origin"])
            start_vec = start_vec / np.linalg.norm(start_vec)
            end_vec = end_vec / np.linalg.norm(end_vec)

            start_angle = angle_from_vector_to_x(start_vec)
            end_angle = angle_from_vector_to_x(end_vec)
            # keep it counter-clockwise first
            if start_angle > end_angle:
                start_angle, end_angle = end_angle, start_angle
                start_vec, end_vec = end_vec, start_vec
            sweep_angle = abs(start_angle - end_angle)

            # # decide direction arc by curve length
            # edge_len = self.c.eval_curveLength(self.did, self.wid, self.eid, edge_id)
            # _len = sweep_angle * radius
            # _len_other = (2 * np.pi - sweep_angle) * radius
            # if abs(edge_len - _len) > abs(edge_len - _len_other):
            #     sweep_angle = 2 * np.pi - sweep_angle
            #     start_vec = end_vec

            # decide direction by middle point
            midpoint = self.c.eval_curve_midpoint(self.did, self.wid, self.eid, edge_id)
            mid_vec = np.array(midpoint) - self.origin
            mid_vec = np.array([np.dot(mid_vec, self.x_axis), np.dot(mid_vec, self.y_axis), np.dot(mid_vec, self.z_axis)])
            mid_vec = mid_vec - np.array(edge_data["param"]["coordSystem"]["origin"])
            mid_vec = mid_vec / np.linalg.norm(mid_vec)
            mid_angle_real = angle_from_vector_to_x(mid_vec)
            mid_angle_now = (start_angle + end_angle) / 2            
            if round(mid_angle_real, 3) != round(mid_angle_now, 3):
                sweep_angle = 2 * np.pi - sweep_angle
                start_vec = end_vec

            ref_vec_dict = xyz_list2dict(list(start_vec))
            curve_dict = OrderedDict({"type": "Arc3D", "start_point": start_point, "end_point": end_point,
                          "center_point": center_point, "radius": radius, "normal": normal,
                          "start_angle": 0.0, "end_angle": sweep_angle, "reference_vector": ref_vec_dict,
                          "curve": edge_id})
        elif edge_type == "Circle":
            # NOTE: There is no vertex id for circle. The origin is the center.
            radius = edge_data["param"]["radius"]
            center_point = xyz_list2dict(edge_data["param"]["coordSystem"]["origin"])
            normal = xyz_list2dict(edge_data["param"]["coordSystem"]["zAxis"])
            curve_dict = OrderedDict({"type": "Circle3D", "center_point": center_point, "radius": radius, "normal": normal,
                          "curve": edge_id})
        else:
            raise NotImplementedError(edge_type, edge_data["vertices"])
        return curve_dict

    def parse_to_fusion360_format(self):
        """parse sketch feature into fusion360 gallery format"""
        name = self.feat_name

        # transform & reference plane
        transform_dict = {"origin": xyz_list2dict(self.plane["origin"]),
                          "z_axis": xyz_list2dict(self.plane["normal"]),
                          "x_axis": xyz_list2dict(self.plane["x"]),
                          "y_axis": xyz_list2dict(list(np.cross(self.plane["normal"], self.plane["x"])))}
        ref_plane_dict = {}

        # profiles
        profiles_dict = {}
        for item in self.geo_topo['faces']:
            # profile level
            profile_id = item['id']
            all_edge_ids = item['edges']
            edge_ids_per_loop = self._parse_edges_to_loops(all_edge_ids)
            #onshapeLogger.debug(f"{edge_ids_per_loop}")
            all_loops = []
            for loop in edge_ids_per_loop:
                if len(loop) == 1:
                    curves = [self._parse_edge_to_fusion360_format(loop[0])]
                else:
                    curves=[self._parse_edge_to_fusion360_format(edge_id) for edge_id in loop]
                loop_dict = {"is_outer": True, "profile_curves": curves}
                all_loops.append(loop_dict)
            profiles_dict.update({profile_id: {"loops": all_loops, "properties": {}}})

        entity_dict = {"name": name, "type": "Sketch", "profiles": profiles_dict,
                       "transform": transform_dict, "reference_plane": ref_plane_dict}
        # onshapeLogger.debug(f"{entity_dict}, name {self.feat_name}")
        return entity_dict



if __name__ == "__main__":
    # with open("test.json","r") as f:
    #     jsonData=json.load(f)
    data_id="00000029"
    link="https://cad.onshape.com/documents/ad34a3f60c4a4caa99646600/w/90b1c0593d914ac7bdde17a3/e/f5cef14c36ad4428a6af59f0"

    v_list = link.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
    c = MyClient(logging=False)

    feature_parser=FeatureListParser(c, did, wid, eid, data_id=data_id)
    save_dict=feature_parser.parse()

    with open("output/parse.json","w") as f:
        json.dump(save_dict, f)