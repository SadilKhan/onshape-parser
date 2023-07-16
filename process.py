import os
import yaml
import json
import numpy as np
from tqdm import tqdm
import argparse
from joblib import delayed, Parallel
from cadparser import FeatureListParser
from myclient import MyClient
from loguru import logger
from rich import print
# create instance of the OnShape client; change key to test on another stack
c = MyClient(logging=False)
from logger import OnshapeParserLogger

onshapeLogger=OnshapeParserLogger().configure_logger().logger


@logger.catch()
def process_one(data_id, link, save_dir):
    save_path = os.path.join(save_dir, "{}.json".format(data_id))
    # if os.path.exists(save_path):
    #     return 1

    v_list = link.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

    # filter data that use operations other than sketch + extrude
    try:
        ofs_data = c.get_features(did, wid, eid).json()
        with open("test.json", "w") as f:
            json.dump(ofs_data, f)
        for item in ofs_data['features']:
            if item['message']['featureType'] not in ['newSketch', 'extrude']:
                #print(data_id,link,item['message']['featureType'])
                return 0
    except Exception as e:
        #print("[{}], contain unsupported features:".format(data_id), e)
        onshapeLogger.error(f"[{data_id}] contains unsupported features. Only Extrusion and Revolutions are supported now.")
        return 0

    # parse detailed cad operations
    # try:
    #     parser = FeatureListParser(c, did, wid, eid, data_id=data_id)
    #     result = parser.parse()
    # except Exception as e:
    #     print("[{}], feature parsing fails:".format(data_id), e)
    #     return 0
    # if len(result["sequence"]) < 2:
    #     return 0
    # with open(save_path, 'w') as fp:
    #     json.dump(result, fp, indent=1)
    # return len(result["sequence"])


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="test with some examples")
parser.add_argument("-p","--link_data_folder", default=None, type=str, help="data folder of onshape links from ABC dataset")
args = parser.parse_args()

if args.test:
    data_examples = {
        "00000002":"https://cad.onshape.com/documents/1ffb81a71e5b402e966b9341/w/6e295017d1b34be684565c40/e/bb398e4615fe4025b34ea8f0" # Has Revolve
        #'00000352': 'https://cad.onshape.com/documents/4185972a944744d8a7a0f2b4/w/d82d7eef8edf4342b7e49732/e/b6d6b562e8b64e7ea50d8325',
                    #'00499992': 'https://cad.onshape.com/documents/051e7a5d1c5847e9582e2135/w/f254b34728601fd9760a2a16/e/b563921eb04567923c3c31d9'
                    #  '00001272': 'https://cad.onshape.com/documents/b53ece83d8964b44bbf1f8ed/w/6b2f1aad3c43402c82009c85/e/91cb13b68f164c2eba845ce6',
                    #  '00001616': 'https://cad.onshape.com/documents/8c3b97c1382c43bab3eb1b48/w/43439c4e192347ecbf818421/e/63b575e3ac654545b571eee6',
                    }
    save_dir = "examples"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for data_id, link in data_examples.items():
        print(data_id)
        process_one(data_id, link, save_dir)

else:
    DWE_DIR = args.link_data_folder
    DATA_ROOT = os.path.dirname(DWE_DIR)
    filenames = sorted(os.listdir(DWE_DIR))
    for name in tqdm(filenames):
        truck_id = name.split('.')[0].split('_')[-1]
        print("Processing truck: {}".format(truck_id))

        save_dir = os.path.join(DATA_ROOT, "processed/{}".format(truck_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dwe_path = os.path.join(DWE_DIR, name)
        with open(dwe_path, 'r') as fp:
            dwe_data = yaml.safe_load(fp)

        total_n = len(dwe_data)
        count = Parallel(n_jobs=10, verbose=2)(delayed(process_one)(data_id, link, save_dir)
                                            for data_id, link in dwe_data.items())
        count = np.array(count)
        print("valid: {}\ntotal:{}".format(np.sum(count > 0), total_n))
        print("distribution:")
        for n in np.unique(count):
            print(n, np.sum(count == n))
