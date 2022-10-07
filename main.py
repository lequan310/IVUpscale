"""
Upscale video: python main.py -v/--video {VIDEO_PATH}
Upscale image: python main.py -v/--video {IMAGE_PATH}
Additional argument:
    -m/--mode [0 or 1] (enhancement mode false or true, enhancement true required opencv installed with cuda)
    -s/--scale value (2, 3, or 4)
Note:
    Up-scaling video takes a while, so be considerate if you want to use enhancement mode
    Feel free to switch or add another Super Resolution model yourself
"""
import cv2
import argparse
import pathlib


def upscale(img):
    up_img = super_res.upsample(img)
    return up_img


def gen_out_path(image):
    file_extension = pathlib.Path(image).suffix
    file_name = pathlib.Path(image).stem
    input_path = pathlib.Path(image).parent.absolute()

    output = '/' + file_name + "_" + str(scale) + "x"
    output += '_enhanced' if enhancement else ""
    output_path = str(input_path) + output + file_extension
    return output_path


def upscale_image(image):
    output_path = gen_out_path(image)
    img = cv2.imread(image)
    height, column, _ = img.shape
    up_img = upscale(img)
    cv2.imwrite(output_path, up_img)
    print('Finish up-scaling image: ' + output_path)


def upscale_video(video):
    output_path = gen_out_path(video)
    cap = cv2.VideoCapture(video)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*scale, height*scale), isColor=True)

    for frame in range(frame_count):
        success, img = cap.read()
        up_img = upscale(img)
        video_writer.write(up_img)
        print('Finish frame ' + str(frame) + '/' + str(frame_count))

    video_writer.release()
    print('Finish up-scaling video: ' + output_path)


parser = argparse.ArgumentParser(description='Provide Image/Video and Scale')
parser.add_argument("-v", "--video", type=str, help="Video path here")
parser.add_argument("-i", "--image", type=str, help="Image path here")
parser.add_argument("-s", "--scale", type=int, help="Video/Image scale (2x, 3x, 4x)")
parser.add_argument("-o", "--output", type=str, help="Output file")
parser.add_argument("-m", "--mode", type=int, help="Enhancement mode 0 or 1 (OFF or ON)")
args = parser.parse_args()

scale = args.scale if 2 <= args.scale <= 4 else 2
enhancement = False if args.mode is None or args.mode == 0 else True
super_res = cv2.dnn_superres.DnnSuperResImpl_create()

# Comment these lines if you're not using opencv with gpu
if enhancement:
    super_res.readModel('EDSR/EDSR_x' + str(scale) + '.pb')
    super_res.setModel('edsr', scale)
else:
    super_res.readModel('ESPCN/ESPCN_x' + str(scale) + '.pb')
    super_res.setModel('espcn', scale)
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    super_res.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    super_res.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if args.video is None and args.image is None:
    print('Provide Image/Video')
elif args.video is not None:
    print('Up-scaling ' + args.video + ' by ' + str(args.scale) + 'x ...')
    upscale_video(args.video)
elif args.image is not None and isinstance(args.scale, int):
    print('Up-scaling ' + args.image + ' by ' + str(args.scale) + 'x ...')
    upscale_image(args.image)

