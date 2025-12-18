from huggingface_hub import HfApi

api = HfApi()

num_keypoints=input("Num of keypoints: ").strip()

num_keypoints=int(num_keypoints)

if num_keypoints not in [17, 133]:
    raise ValueError(f"Num of keypoints {num_keypoints} not in [17, 133]")

if num_keypoints == 17:
    token = ""
else:
    token = ""

api.upload_file(
    path_or_fileobj=f"net_{num_keypoints}_best.pth",  # Local path to your file
    path_in_repo=f"rtm{num_keypoints}lifting.pth",                    # Name it should have in the Space
    repo_id=f"fhswf/rtm{num_keypoints}lifting",
    token=token,
)

print("Upload successful!")