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

    # NOTE: modify below to play with query
    if args.algo in ["cfusion", "vlfusion"]:
        # points, colors = slam.query("Window", topk=3)
        # points, colors = slam.query("there is a stainless steel fridge in the ketchen", topk=3)
        # points, colors, labels = slam.semantic_query([
        #     "basket", "blanket", "blinds", "book", "cabinet", "candle", "chair", "cushion",
        #     "ceiling", "door", "floor", "indoor-plant", "lamp", "picture", "pillar", "plant-stand",
        #     "plate", "pot", "sofa", "stool", "switch", "table", "vase", "vent", "wall", "wall-plug",
        #     "window", "rug"
        # ])#room0
        # points, colors, labels = slam.semantic_query([
        #     "basket", 'bed', "blanket", "blinds", "book", "cabinet", 'comforter',
        #     "ceiling", "door", "floor", "indoor-plant", "lamp",'nightstand', 'panel', "picture", 
        #     'pillow'
        #     "plate", "switch",  "vase", "vent", "wall", "wall-plug",
        #     "window", "rug"
        # ])#room1
        # points, colors, labels = slam.semantic_query([
        #     "blinds", "bottle", 'box','bowl','chair',
        #     "ceiling", "door", "floor", "indoor-plant", "lamp", 'plate', 'sculpture','shelf',"switch", 'table', "vase", "vent", "wall", "wall-plug",
        #     "window", "rug"
        # ])#room2
        # points, colors, labels = slam.semantic_query([
        #     'bin','blinds', 'camera', 'chair', 'clock', 'ceiling', 
        #     'desk-organizer', 'door', 'floor', 'indoor-plant', 'lamp', 'panel', 
        #     'pillar', 'plant-stand', 'sofa', 'switch', 'table', 'tablet', 'tissue-paper', 
        #     'tv-screen', 'vent', 'wall', 'wall-plug', 'rug'
        # ])#office0
        # points, colors, labels = slam.semantic_query([
        #     'bin', 'blanket', 'blinds', 'bottle', 'chair', 'cloth', 'ceiling', 'desk',
        #     'desk-organizer', 'door', 'floor', 'lamp', 'monitor', 'panel', 'pillar', 
        #     'pillow', 'switch', 'table', 'tissue-paper', 'vent', 'wall', 'wall-plug'
        # ])#office1
        # points, colors, labels = slam.semantic_query([
        #     'bin', 'bottle', 'camera', 'chair', 'clock', 'cushion', 'ceiling', 'door', 'floor', 
        #     'lamp', 'panel', 'sofa', 'stool', 'table', 'tablet', 'tissue-paper', 'tv-screen', 
        #     'vent', 'wall', 'wall-plug', 'window'
        # ])#office2
        # points, colors, labels = slam.semantic_query([
        #     'bench','bin','blinds', 'bottle','box', 'camera', 'chair', 'clock', 'cushion', 'ceiling', 'desk-organizer',
        #       'door', 'floor', 'lamp', 'panel', 'pipe', 'sofa', 'switch', 'table', 'tablet', 'tissue-paper', 'tv-stand', 'vent', 'wall', 
        #       'wall-plug', 'window'
        # ])#office3
        # points, colors, labels = slam.semantic_query([
        # 'bench', 'bin', 'camera', 'chair', 'clock', 'ceiling', 'door', 'floor', 'lamp', 'panel', 'table', 'tv-screen', 'vent', 'wall', 'window'
        # ])#office4
        # points, colors, labels = slam.semantic_query([
        #     "wall", "floor", "cabinet", "chair", "table",
        #      "door","window" , "counter","refrigerator", "sink"
        # ])#scannet 0011
        # points, colors, labels = slam.semantic_query([
        #     "wall", "floor", "chair", "table",
        #      "door","window" ,  'bookshelf'
        # ])#scannet 0030
        # points, colors, labels = slam.semantic_query([
        #     "wall", "floor", 'bed', "chair", "table",
        #      "door",'picture','desk', 'curtain', 'shower curtain', 'toilet', "sink", 'bathtub'
        # ])#scannet 0046
        # points, colors, labels = slam.semantic_query([
        #     "wall", "floor",  "door",'window', 'toilet', "sink", 
        # ])#scannet 0086
        # points, colors, labels = slam.semantic_query([
        #     "wall", "floor", 'bed', "chair",
        #      "door",'window','desk', 'refridgerator'
        # ])#scannet 0222
        # points, colors, labels = slam.semantic_query([
        #     "wall", "floor", 'cabinet', 'bed', "chair",'table',
        #      "door", 'picture','curtain', 'refridgerator'
        # ])#scannet 0389
        # points, colors, labels = slam.semantic_query([
        #     "wall", "floor", 'cabinet', "chair",
        #      "door",'window','bookshelf','desk'
        # ])#scannet 0378
        points, colors, labels = slam.semantic_query([
            "wall", "floor", 'cabinet', 'bed', "chair", 
             "door",'picture','desk', 'curtain','refridgerator', 'shower curtain', 'toilet', "sink", 'bathtub'
        ])#scannet 0435
        # [(0, 'wall'), (1, 'floor'), (2, 'cabinet'), (4, 'chair'), (6, 'table'), (7, 'door'), (8, 'window'), (11, 'counter'), (14, 'refridgerator'), (17, 'sink')]
        # [(0, 'wall'), (1, 'floor'), (4, 'chair'), (6, 'table'), (7, 'door'), (8, 'window'), (9, 'bookshelf')]
        # [(0, 'wall'), (1, 'floor'), (3, 'bed'), (4, 'chair'), (6, 'table'), (7, 'door'), (10, 'picture'), (12, 'desk'), (13, 'curtain'), (15, 'shower curtain'), (16, 'toilet'), (17, 'sink'), (18, 'bathtub')]
        # [(0, 'wall'), (1, 'floor'), (7, 'door'), (8, 'window'), (16, 'toilet'), (17, 'sink')]
        #  [(0, 'wall'), (1, 'floor'), (3, 'bed'), (4, 'chair'), (7, 'door'), (8, 'window'), (12, 'desk'), (14, 'refridgerator')
        # [(0, 'wall'), (1, 'floor'), (2, 'cabinet'), (3, 'bed'), (4, 'chair'), (6, 'table'), (7, 'door'), (10, 'picture'), (13, 'curtain'), (14, 'refridgerator')
            # [(0, 'wall'), (1, 'floor'), (2, 'cabinet'), (4, 'chair'), (7, 'door'), (8, 'window'), (9, 'bookshelf'), (12, 'desk')]
        # (0, 'wall'), (1, 'floor'), (2, 'cabinet'), (3, 'bed'), (4, 'chair'), (7, 'door'), (10, 'picture'), (12, 'desk'), (13, 'curtain'), (14, 'refridgerator'), (15, 'shower curtain'), (16, 'toilet'), (17, 'sink'), (18, 'bathtub')
        points, colors = slam.query(query="find the table that is far away from the sink", topk=3)
        # show_pc(points, colors, slam.point_state.poses)
        # save_pc(points, colors, f"{args.data}_{args.scene}/semantic_pc.ply")
        np.savez_compressed(f"{args.data}_{args.scene}/openfusion_vlfusion", points=points, labels=labels)
        
if __name__ == "__main__":
    main()