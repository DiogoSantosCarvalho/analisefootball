import cv2

def read_video(vid_path, frame_count=300):
    """This function reads a video file and yields each frame of the video"""

    frames = []
    cap = cv2.VideoCapture(vid_path)
    counter = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frames.append(frame)
        else:
            break

        counter += 1
        if counter == frame_count:
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames


def get_video_fps(vid_path):
    """This function returns the FPS of a video file"""
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25


def write_video(frames, out_path, fps=25):
    """This function writes the frames to a video file at the best possible quality.
    
    Tries codecs in order of quality preference:
    1. avc1 (H.264) - best quality/size ratio
    2. mp4v (MPEG-4) - reliable fallback
    """
    height, width, _ = frames[0].shape

    # Try codecs in order of quality preference
    codecs = [('avc1', out_path), ('mp4v', out_path)]
    out = None

    for codec, path in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Using codec: {codec} | Resolution: {width}x{height} | FPS: {fps}")
            out = writer
            break
        writer.release()

    if out is None:
        raise RuntimeError("No suitable video codec found. Install H.264 support or check OpenCV installation.")

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def show_image(image, title="Image"):
    """This function displays an image"""

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()