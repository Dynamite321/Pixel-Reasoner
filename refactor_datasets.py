import os
from datasets import load_dataset

datasets = {
    'sft': 'TIGER-Lab/PixelReasoner-SFT-Data',
    'rl': 'TIGER-Lab/PixelReasoner-RL-Data',
}

paths = {
    'sft': '/fsx-project/yanjunfu/datasets/Pixel-Reasoner/sft',
    'rl': '/fsx-project/yanjunfu/datasets/Pixel-Reasoner/rl',
}

def main():
    for dataset_name, dataset_path in paths.items():
        dataset = load_dataset(datasets[dataset_name])['train']
        if dataset_name == 'sft':
            for entry in dataset['message_list']:
                for turn in entry:
                    for content in turn['content']:
                        if content['image'] is not None:
                            image_path = os.path.join(dataset_path, content['image'])
                            if not os.path.exists(image_path):
                                print(f"Image not found: {image_path}")
                                continue
                            content['image'] = image_path
                    
                    for content in turn['content']:
                        if content['video'] is not None:
                            for i in range(len(content['video'])):
                                video_path = os.path.join(dataset_path, content['video'][i])
                            if not os.path.exists(video_path):
                                print(f"Video not found: {video_path}")
                                continue
                            content['video'][i] = video_path
        else:
            for entry in dataset['image']:
                # if image is a list, then it is a video
                if isinstance(entry, list):
                    for i, image in enumerate(entry):
                        image_path = os.path.join(dataset_path, image)
                        if not os.path.exists(image_path):
                            print(f"Image not found: {image_path}")
                            continue
                        entry[i] = image_path
                else:
                    image_path = os.path.join(dataset_path, entry)
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        continue
                    entry = image_path
        
        dataset.to_parquet(os.path.join(dataset_path, 'train.parquet'))

if __name__ == "__main__":
    main()