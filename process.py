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
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
import multiprocessing
#multiprocessing.set_start_method("forkserver",force=True)

# create instance of the OnShape client; change key to test on another stack
c = MyClient(logging=False)
from logger import OnshapeParserLogger

onshapeLogger=OnshapeParserLogger().configure_logger(verbose=False).logger


@logger.catch()
def process_one(data_id, link, save_dir):
    save_path = os.path.join(save_dir, "{}.json".format(data_id))
    # if os.path.exists(save_path):
    #     return 1

    v_list = link.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

    # filter data that use operations other than sketch + extrude
    # try:
    #     ofs_data = c.get_features(did, wid, eid).json()
    #     with open("test.json", "w") as f:
    #         json.dump(ofs_data, f)
    #     for item in ofs_data['features']:
    #         if item['message']['featureType'] not in ['newSketch', 'extrude', "revolve"]:
    #             print(data_id,link,item['message']['featureType'])
    #             return 0
    # except Exception as e:
    #     #print("[{}], contain unsupported features:".format(data_id), e)
    #     onshapeLogger.error(f"[{data_id}] contains unsupported features. Only Extrusion and Revolutions are supported now. {e}")
    #     return 0

    # parse detailed cad operations
    try:
        parser = FeatureListParser(c, did, wid, eid, data_id=data_id)
        result = parser.parse()
    except Exception as e:
        print("[{}], feature parsing fails:".format(data_id), e)
        return 0
    if len(result["sequence"]) < 2:
        return 0
    with open(save_path, 'w') as fp:
        json.dump(result, fp, indent=1)
    return len(result["sequence"])


def process_yaml(name,data_root,dwe_dir,max_workers):
    truck_id = name.split('.')[0].split('_')[-1]
    print("Processing truck: {}".format(truck_id))

    save_dir = os.path.join(data_root, "processed/{}".format(truck_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dwe_path = os.path.join(dwe_dir, name)
    with open(dwe_path, 'r') as fp:
        dwe_data = yaml.safe_load(fp)

    total_n = len(dwe_data)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, *item, save_dir): item for item in tqdm(dwe_data.items())}
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            item = futures[future]
            result = future.result()
            results.append(result)

    # count = np.array(results)
    # print("valid: {}\ntotal:{}".format(np.sum(count > 0), total_n))
    # print("distribution:")
    # for n in np.unique(count):
    #     print(n, np.sum(count == n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="test with some examples")
    parser.add_argument("-p","--link_data_folder", default=None, type=str, help="data folder of onshape links from ABC dataset")
    parser.add_argument("-d","--dwe_folder", default=None, type=str, help="data folder of dwe files from ABC dataset")
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--end",type=int,default=10)
    parser.add_argument("--max_workers",type=int,default=32)
    args = parser.parse_args()

    if args.test:
        data_examples = {
            #"00000029":"https://cad.onshape.com/documents/ad34a3f60c4a4caa99646600/w/90b1c0593d914ac7bdde17a3/e/f5cef14c36ad4428a6af59f0" # Revolve
            # "00000016":"https://cad.onshape.com/documents/b08aa818955948c690fd9b6d/w/abe349c63cc94246bf308723/e/bXCQKPgEejPohcpU9rBMvL53", # Fillet
            # "00000031":"https://cad.onshape.com/documents/ad34a3f60c4a4caa99646600/w/90b1c0593d914ac7bdde17a3/e/w0LpLSvmnVWF4omQ66tVspot", # Circular Pattern
            # "00000015":"https://cad.onshape.com/documents/b08aa818955948c690fd9b6d/w/abe349c63cc94246bf308723/e/48b61785c4f64313a22ba758", # Shell
            # "00000028":"https://cad.onshape.com/documents/ad34a3f60c4a4caa99646600/w/90b1c0593d914ac7bdde17a3/e/og81BIAlwU3qxwrYDIgLKJhJ", # Draft
            # "00000005":"https://cad.onshape.com/documents/d4fe04f0f5f84b52bd4f10e4/w/af184e4c3083411ba6f2afac/e/da756952509a495bb53a1aae", # LinearPattern
            #"00000011":"https://cad.onshape.com/documents/e909f412cda24521865fac0f/w/6f8b499942424a50a940c5f6/e/50bc16864ff74c1280f3d506", # cPlane
            #"0000007": "https://cad.onshape.com/documents/767e4372b5f94a88a7a17d90/w/194c02e4f65d47dabd006030/e/fc1b493ec8b197f5902934c9" # Chamfer
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

        for name in tqdm(filenames[args.start:args.end]):
            process_yaml(name,data_root=DATA_ROOT,dwe_dir=DWE_DIR,max_workers=args.max_workers)



if __name__=="__main__":
    main()