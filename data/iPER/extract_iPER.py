import cv2
import argparse
from os import path, makedirs
import pickle
from tqdm import tqdm
# import imagesize
from glob import glob
from natsort import natsorted

def get_image(vidcap, frame_number,spatial_size=None):
    vidcap.set(1, frame_number)
    _, img = vidcap.read()
    if spatial_size is not None and spatial_size != img.shape[0]:
        img=cv2.resize(img,(spatial_size,spatial_size),interpolation=cv2.INTER_LINEAR)
    return img


def process_video(f_name, args):


    # open video
    base_raw_dir = args.raw_dir.split("*")[0]
    fn = f_name
    vid_path = path.join(base_raw_dir, fn)
    # vid_path = f"Code/input/train_data/movies/{fn}"
    vidcap = cv2.VideoCapture()
    vidcap.open(vid_path)
    counter = 0
    while not vidcap.isOpened():
        counter += 1
        time.sleep(1)
        if counter > 10:
            raise Exception("Could not open movie")

    # get some metadata
    number_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # create target path if not existent
    base_path = path.join(args.processed_dir, fn.split(".")[0])
    makedirs(base_path, exist_ok=True)


    # begin extraction
    for frame_number in tqdm(range(0, number_frames),desc=f'Processing overall {number_frames} individual frames'):
        image_target_file = path.join(base_path, f"frame_{frame_number}.png")
        # FRAME
        if not path.exists(image_target_file):
            # write frame itself
            img = get_image(vidcap, frame_number)
            if img is None:
                continue
            try:
                if args.spatial_size is None:
                    success = cv2.imwrite(image_target_file, img)
                else:
                    img_res = cv2.resize(img,(args.spatial_size,args.spatial_size), interpolation=cv2.INTER_LINEAR)
                    success = cv2.imwrite(image_target_file,img_res)
            except cv2.error as e:
                print(e)
                continue
            except Exception as ex:
                print(ex)
                continue

def extract(args):
    base_dir = args.raw_dir
    data_names = [p.split(base_dir)[-1] for p in glob(path.join(args.raw_dir,'*.mp4')) ]

    for dn in tqdm(data_names, desc='Extracting video files.'):
        process_video(dn,args)

    prepare(args)



def prepare(args):

    datadict = {
        "img_path": [],
        "fid": [],
        "vid": [],
        "img_size": [],
        "object_id":[],
        'action_id': [],
        'actor_id': [],
    }

    videos = [d for d in glob(path.join(args.processed_dir, "*")) if path.isdir(d)]

    videos = natsorted(videos)

    for vid, vid_name in enumerate(videos):

        images = glob(path.join(vid_name, "*.png"))
        images = natsorted(images)



        object_id = 100 * int(vid_name.split("/")[-1].split("_")[0]) + int(vid_name.split("/")[-1].split("_")[1])
        actor_id = int(vid_name.split("/")[-1].split("_")[0])
        action_id = int(vid_name.split("/")[-1].split("_")[-1])


        for i, img_path in enumerate(
                tqdm(
                    images,
                    desc=f'Extracting meta information of video "{vid_name.split("/")[-1]}"',
                )
        ):
            fid = int(img_path.split("_")[-1].split(".")[0])
            img_path_rel = img_path.split(args.processed_dir)[1]

            breakpoint()
            if args.spatial_size is None:
                w_img, h_img = imagesize.get(img_path)
            else:
                w_img = args.spatial_size
                h_img = args.spatial_size


            datadict["img_path"].append(img_path_rel)
            datadict["fid"].append(fid)
            datadict["vid"].append(vid)
            datadict["img_size"].append((h_img, w_img))
            datadict["object_id"].append(object_id)
            datadict["action_id"].append(action_id)
            datadict["actor_id"].append(actor_id)


    # Store data (serialize)
    save_path = path.join(
        args.processed_dir, f"{args.meta_file_name}.p"
    )
    with open(save_path, "wb") as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":

    import time
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir",
        "-v",
        type=str,
        default="INSERT PATH",
    )
    parser.add_argument(
        "--processed_dir",
        "-p",
        type=str,
        default="INSERT PATH",
    )
    parser.add_argument("--meta_file_name","-mfn",type=str,default="meta_data", help="The name for the pickle file, where the meta data is stored (without ending).")
    parser.add_argument("--video_format", "-vf", type=str, default="mp4", choices=["mkv","mp4"],help="Format of the input videos to the pipeline.")
    parser.add_argument("--spatial_size", "-s",type=int, default=256,help="The desired spatial_size of the output.")
    parser.add_argument("--input_size", "-i", type=int, default=1024, help="The spatial_size of the input videos.")
    args = parser.parse_args()

    extract(args)

