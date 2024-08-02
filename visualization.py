import open3d as o3d
import numpy as np

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False):
    """
    Credit: https://github.com/delestro/rand_cmap/tree/master
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    Args:
        nlabels: Number of labels (size of colormap)
        type: 'bright' for strong colors, 'soft' for pastel colors
        first_color_black: Option to use first color as black, True or False
        last_color_black: Option to use last color as black, True or False
    Usage:
        cmap = rand_cmap(155)
        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(
            np.array([cmap(i) for i in keys.cpu().numpy()])[:,:3]
        )
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [
            [np.random.uniform(low=0.0, high=1),
             np.random.uniform(low=0.2, high=1),
             np.random.uniform(low=0.9, high=1)
         ] for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [
            [np.random.uniform(low=low, high=high),
             np.random.uniform(low=low, high=high),
             np.random.uniform(low=low, high=high)
             ] for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap

def get_cmap_legend(cmap, labels, row_length=10, savefile=None):
    """ generate legend for colormap

    Args:
        cmap (matplotlib.colors.Colormap): colormap
        labels (list): list of class names
        row_length (int, optional): number of classes in one row. Defaults to 10.
    """
    import matplotlib.patheffects as pe
    from matplotlib import pyplot as plt

    colors = [cmap(i) for i in range(len(labels))]
    max_text_length = max(len(label) for label in labels)
    # NOTE: adjust the figure size based on the maximum text length
    plt.figure(figsize=(row_length * max_text_length * 0.2, (len(labels) + row_length - 1) // row_length))

    for i, (color, class_name) in enumerate(zip(colors, labels)):
        row = i // row_length
        col = i % row_length
        plt.fill_between([col, col + 1], -row, -row + 1, color=color)
        plt.text(
            col + 0.5, -row + 0.5, class_name, ha='center', va='center',
            rotation=0, color='white',
            path_effects=[pe.withStroke(linewidth=3, foreground='black')]
        )

    plt.axis('off')
    plt.ylim(-(len(labels) + row_length - 1) // row_length, 1)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

def get_color_map_legend(colors, labels, row_length=10, savefile=None):
    """ generate legend for colormap

    Args:
        cmap (matplotlib.colors.Colormap): colormap
        labels (list): list of class names
        row_length (int, optional): number of classes in one row. Defaults to 10.
    """
    import matplotlib.patheffects as pe
    from matplotlib import pyplot as plt

    max_text_length = max(len(label) for label in labels)
    # NOTE: adjust the figure size based on the maximum text length
    plt.figure(figsize=(row_length * max_text_length * 0.2, (len(labels) + row_length - 1) // row_length))
    color_map = {
        'wall':0,
        "floor":1,
        "cabinet":2,
        "chair":4,
        "table":6,
        "door":7,
        "window":8,
        "counter":11,
        "refridgerator":14,
        "sink":17,
        "other":19
    }
    
    for i, (color, class_name) in enumerate(zip(colors, labels)):
        row = i // row_length
        col = i % row_length
        plt.fill_between([col, col + 1], -row, -row + 1, color=colors[color_map[class_name]])
        plt.text(
            col + 0.5, -row + 0.5, class_name, ha='center', va='center',
            rotation=0, color='white',
            path_effects=[pe.withStroke(linewidth=3, foreground='black')]
        )

    plt.axis('off')
    plt.ylim(-(len(labels) + row_length - 1) // row_length, 1)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

def get_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    print(type(points))
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.array(colors)
    print(type(colors))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def show_pc(points, colors, poses=None):
    pcd = get_pcd(points, colors)
    if poses is not None:
        cameras_list = []
        for pose in poses:
            camera_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            cameras_list.append(camera_cf.transform(np.linalg.inv(pose)))
        o3d.visualization.draw_geometries([pcd, *cameras_list])
    else:
        o3d.visualization.draw_geometries([pcd])


import numpy as np
from matplotlib.colors import hsv_to_rgb

def generate_distinct_colors(n_colors=28):
    """
    Generates n distinct colors.
    
    Args:
        n_colors (int): Number of distinct colors to generate.
    
    Returns:
        list: List of RGB tuples.
    """
    # Generate colors in HSV space
    hsv_colors = [(i / n_colors, 1.0, 1.0) for i in range(n_colors)]
    
    # Convert to RGB
    rgb_colors = [list(hsv_to_rgb(color)) for color in hsv_colors]
    
    return rgb_colors

def generate_colors():
    
    
    
    
    
    
    color_map =  [(0, 0, 0),#
       (9, 7, 230),		#
       (152, 223, 138),		# 
       (31, 119, 180), 		
       (255, 187, 120),		# 
       (255, 255, 0), 		 
       (214, 39, 40),  		# 
       (197, 176, 213),		#
      (112, 9, 255),	#
       (196, 156, 148),		
       (23, 190, 207), 		
        (255, 255, 52),  ##
       (247, 182, 210),		 
       (66, 188, 102), 
       (255, 0, 255),#
       (255, 0, 255), 
       (202, 185, 52),
       (51, 255, 255),# 
       (200, 54, 131), 
       (10, 255, 71),  #
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		
       (112, 128, 144),		
       (96, 207, 209), 
       (227, 119, 194),		
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		
       (100, 85, 144)]
    rgb_colors = np.array([list(color_map[color]) for color in range(39)])/255
    
    return rgb_colors.tolist()

# Example usage
colors = generate_distinct_colors(28)
for i, color in enumerate(colors):
    print(f"Color {i+1}: {color}")

PATH_POINTCLOUD = "replica_room0/points_labels.npz"
# PATH_vlfusion = "replica_room0/vlfusion.npz"
def main():

    loaded = np.load(PATH_POINTCLOUD)
    # vlfusion = np.load(PATH_vlfusion)
    # print(vlfusion['attr_name_color'])
    
    query = np.unique(loaded['labels'])
    print(query)
    # text_labels = [
    #         "basket", "blanket", "blinds", "book", "cabinet", "candle", "chair", "cushion",
    #         "ceiling", "door", "floor", "indoor-plant", "lamp", "picture", "pillar", "plant-stand",
    #         "plate", "pot", "sofa", "stool", "switch", "table", "vase", "vent", "wall", "wall-plug",
    #         "window", "rug", "other"
    #     ]
    
    text_labels =[
            "wall", "floor", "cabinet", "chair", "table",
             "door","window" , "counter","refridgerator", "sink", "other"
        ]
    cmap = rand_cmap(len(text_labels), type="soft", first_color_black=False)

    import json

    colors = cmap(np.linspace(0, 1, 40))
    colors_list = generate_colors()
    print(colors_list)
    with open("color_map.json", 'w') as f:
        json.dump(colors_list, f)
    get_color_map_legend(colors_list, text_labels, savefile="color_map.png")
    # get_cmap_legend(cmap, text_labels, savefile="color_map.png")
    # print(list(cmap(0)))
    colors = [list(cmap(label))[:3] for label in loaded['labels']]
    points = loaded['points']
    show_pc(points, colors)

if __name__ == "__main__":
    main()