import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import open3d as o3d
from openfusion.slam import build_slam, BaseSLAM
from openfusion.datasets import Dataset
from openfusion.utils import (
    show_pc, save_pc, get_cmap_legend
)
from configs.build import get_config

import json
def stream_loop(args, slam:BaseSLAM):
    if args.save:
        slam.export_path = f"{args.data}_live/{args.algo}.npz"

    slam.start_thread()
    if args.live:
        slam.start_monitor_thread()
        slam.start_query_thread()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        slam.stop_thread()
        if args.live:
            slam.stop_query_thread()
            slam.stop_monitor_thread()


def dataset_loop(args, slam:BaseSLAM, dataset:Dataset):
    if args.save:
        slam.export_path = f"{args.data}_{args.scene}_{args.algo}.npz"

    if args.live:
        slam.start_monitor_thread()
        slam.start_query_thread()
    i = 0
    for rgb_path, depth_path, extrinsics in tqdm(dataset):
        rgb, depth = slam.io.from_file(rgb_path, depth_path)
        slam.io.update(rgb, depth, extrinsics)
        slam.vo()
        slam.compute_state(encode_image=i%10==0)
        i += 1
    if args.live:
        slam.stop_query_thread()
        slam.stop_monitor_thread()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
    parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
    parser.add_argument('--data', type=str, default="kobuki", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="icra", help='Name of the scene in the dataset.')
    parser.add_argument('--frames', type=int, default=-1, help='Total number of frames to use. If -1, use all frames.')
    parser.add_argument('--device', type=str, default="cuda:0", choices=["cpu:0", "cuda:0"])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--host_ip', type=str, default="YOUR IP") # for stream
    args = parser.parse_args()

    if args.stream:
        args.scene = "live"
        if not os.path.exists(f"sample/{args.data}"):
            os.mkdir(f"sample/{args.data}")
            raise ValueError(f"[*] please place the intrinsic.txt inside `sample/{args.data}/`.")
        if not os.path.exists(f"sample/{args.data}/live"):
            os.mkdir(f"sample/{args.data}/live")

    params = get_config(args.data, args.scene)
    dataset:Dataset = params["dataset"](params["path"], args.frames, args.stream)
    intrinsic = dataset.load_intrinsics(params["img_size"], params["input_size"])
    print(params, args)
    slam = build_slam(args, intrinsic, params)

    # NOTE: real-time semantic map construction
    if not os.path.exists(f"{args.data}_{args.scene}"):
        os.makedirs(f"{args.data}_{args.scene}")
    if args.load:
        if os.path.exists(f"{args.data}_{args.scene}/{args.algo}.npz"):
            print("[*] loading saved state...")
            slam.point_state.load(f"{args.data}_{args.scene}/{args.algo}.npz")
        else:
            print("[*] no saved state found, skipping...")
    else:
        if args.stream:
            stream_loop(args, slam)
        else:
            dataset_loop(args, slam, dataset)
            if args.save:
                slam.save(f"{args.data}_{args.scene}/{args.algo}.npz")

    # NOTE: save point cloud
    points, colors = slam.point_state.get_pc()
    save_pc(points, colors, f"{args.data}_{args.scene}/color_pc.ply")

    # NOTE: save colorized mesh
    mesh = slam.point_state.get_mesh()
    o3d.io.write_triangle_mesh(f"{args.data}_{args.scene}/color_mesh.ply", mesh)
    o3d.io.write_triangle_mesh(f"{args.data}_{args.scene}/color_mesh.glb", mesh)

    # points, colors = slam.query(query="find the trash can that is close to the doors", topk=3)
    # show_pc(points, colors)
    with open(f"nr3d/{args.scene}_annotation.json", 'r') as f:
        data = json.load(f)
    # NOTE: modify below to play with query
    ans = []
    bbos = []
    """
    {"id", "utterance", "bbox_extent", "bbox_center"}
    """
    if args.algo in ["cfusion", "vlfusion"]:
        for d in data:
            try:      
                points = slam.query(query=d["utterance"], topk=3, only_poi=True)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                # print(points)
                if len(points) != 0:
                    bbo = pcd.get_oriented_bounding_box()
                    bbos.append(bbo)
                    # print(bbos[-1])
                    ans.append(
                        {
                            "id": d["target_id"],
                            "utterance":d["utterance"],
                            "bbox_extent":np.copy(bbos[-1].extent).tolist(),
                            "bbox_center":np.copy(bbos[-1].center).tolist()
                        }
                    )
                else:
                    ans.append(
                        {
                            "id": d["target_id"],
                            "utterance":d["utterance"],
                            "bbox_extent":[0, 0, 0],
                            "bbox_center":[0, 0, 0]
                        }
                    )
            except:
                ans.append(
                        {
                            "id": d["target_id"],
                            "utterance":d["utterance"],
                            "bbox_extent":[0, 0, 0],
                            "bbox_center":[0, 0, 0]
                        }
                    )
    with open(f"nr3d/OpenFusion/{args.scene}.json", 'w') as out:
        json.dump(ans, out)
if __name__ == "__main__":
    main()