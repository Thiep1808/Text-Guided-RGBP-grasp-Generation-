import os
import sys
import json
import spacy  # for tokenizer
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy.io as scio
from torch.utils.data import Dataset
import collections.abc as container_abcs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points

spacy_eng = spacy.load("en_core_web_sm")


class Text2GraspDataset(Dataset):
    def __init__(self, root, start_scene, end_scene, test_split=True, seq_max_len=50, camera='kinect', split='train', num_points=20000, freq_threshold=1,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible

        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}

        self.seq_max_len = seq_max_len

        if split == 'train':
            self.sceneIds = list(range(start_scene, end_scene))
        elif split == 'test':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_seen':
            self.sceneIds = list(range(140, 141))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.read_vocab(f'{root}/text_data/vocab.json')
        text_path = open(f'{root}/text_data/data.json', 'r')
        self.text_data = json.load(text_path)

        self.frameid = []
        self.super_paths = []
        object_it = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            data_path = os.path.join(root, 'scenes', f'scene_{x}', 'object_id_list.txt')
            with open(data_path, 'r') as file:
                objects_id = file.readlines()
            for object_id in objects_id:
                object_id = int(object_id.strip()) + 1
                if object_id in self.objectid() or object_id==20:
                    object_it.append(int(object_id))
                    for img_num in range(256):
                        for i in range(len(self.text_data[str(object_id)])):
                            super_path = f'{object_id}_{i}_{x}_{str(img_num)}'
                            self.super_paths.append(super_path)
                            self.frameid.append(int(img_num))

            if self.load_label:
                collision_labels = np.load(
                    os.path.join(root, 'collision_label', f'scene_{x.strip()}', 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]
        if test_split:
            self.valid_obj_idxs, self.grasp_labels = self.load_grasp_labels(set(object_it))
        else:
            self.valid_obj_idxs, self.grasp_labels = None, None

    def scene_list(self):
        return set([i.split('_')[2] for i in self.super_paths])

    def __len__(self):
        return len(self.super_paths)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def path(self, index):
        return self.super_paths[index]

    def __getitem__(self, index):
        obj_id, index_text, sceneId, img_num = self.super_paths[index].split('_')
        descript = self.text_data[obj_id][int(index_text)]
        numericalized_descript = self.get_text_data(descript)
        if self.load_label:
            grasp_data = self.get_data_label(index, int(obj_id), sceneId, img_num)
        else:
            grasp_data = self.get_data(index, sceneId, img_num)

        grasp_data['text_input'] = np.array(numericalized_descript)

        return grasp_data

    def get_text_data(self, descript):

        numericalized_descript = [self.vocab.stoi["<SOS>"]]
        numericalized_descript += self.vocab.numericalize(descript)
        numericalized_descript.append(self.vocab.stoi["<EOS>"])

        padding = [self.vocab.stoi["<PAD>"] for _ in range(self.seq_max_len - len(numericalized_descript))]
        numericalized_descript.extend(padding)
        return torch.tensor(numericalized_descript)

    def get_data(self, index, sceneId, img_num, return_raw_cloud=False):
        color_path = os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'rgb',
                                  img_num.zfill(4) + '.png')
        depth_path = os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'depth',
                                  img_num.zfill(4) + '.png')
        label_path = os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'label',
                                  img_num.zfill(4) + '.png')
        meta_path = os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'meta',
                                 img_num.zfill(4) + '.mat')
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(label_path))
        meta = scio.loadmat(meta_path)
        scene = sceneId
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        cloud_sampled = np.concatenate((cloud_sampled, color_sampled), axis=1)
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)

        return ret_dict

    def get_data_label(self, index, obj_id, sceneId, img_num):
        color_path = os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'rgb',
                                  img_num.zfill(4) + '.png')
        depth_path = os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'depth',
                                  img_num.zfill(4) + '.png')
        label_path = os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'label',
                                  img_num.zfill(4) + '.png')
        meta_path = os.path.join(self.root, 'scenes', f'scene_{sceneId}', self.camera, 'meta',
                                 img_num.zfill(4) + '.mat')
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(label_path))
        meta = scio.loadmat(meta_path)
        scene = sceneId
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', f'scene_{scene}', self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', f'scene_{scene}', self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label != obj_id] = 0
        objectness_label[objectness_label == obj_id] = 1
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []
        if (seg_sampled == obj_id).sum() < 50 or obj_id == 19:
            objectness_label[objectness_label == obj_id] = 0

        object_poses_list.append(poses[:, :, list(obj_idxs).index(obj_id)])
        points, offsets, scores, tolerance = self.grasp_labels[obj_id]
        collision = self.collision_labels[scene][list(obj_idxs).index(obj_id)]  # (Np, V, A, D)
        # print(obj_id)
        # remove invisible grasp points

        if self.remove_invisible:
            visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_id], points,
                                                         poses[:, :, list(obj_idxs).index(obj_id)], th=0.01)
            points = points[visible_mask]
            offsets = offsets[visible_mask]
            scores = scores[visible_mask]
            tolerance = tolerance[visible_mask]
            collision = collision[visible_mask]

        idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)

        grasp_points_list.append(points[idxs])
        grasp_offsets_list.append(offsets[idxs])
        collision = collision[idxs].copy()
        scores = scores[idxs].copy()
        scores[collision] = 0
        grasp_scores_list.append(scores)
        tolerance = tolerance[idxs].copy()
        tolerance[collision] = 0
        grasp_tolerance_list.append(tolerance)

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        cloud_sampled = np.concatenate((cloud_sampled, color_sampled), axis=1)

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        # ret_dict['rgb'] = rgb.numpy()
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list

        return ret_dict


    def load_grasp_labels(self, oject_it):
        obj_ids = oject_it
        valid_obj_idxs = []  # start 1
        grasp_labels = {}
        for obj_id in tqdm(obj_ids, desc='Loading grasping labels...'):
            if obj_id == 19: continue
            valid_obj_idxs.append(obj_id)  # here align with label png
            label = np.load(os.path.join(self.root, 'grasp_label', '{}_labels.npz'.format(str(obj_id - 1).zfill(3))))
            tolerance = np.load(
                os.path.join(BASE_DIR, 'tolerance', '{}_tolerance.npy'.format(str(obj_id - 1).zfill(3))))
            grasp_labels[obj_id] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                    label['scores'].astype(np.float32), tolerance)

        return valid_obj_idxs, grasp_labels

    def objectid(self):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 33, 35,
                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69,
                      71, 72, 75]


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


class Vocabulary:
    def __init__(self, freq_threshold=5):
        # Initialize 2 dictionary: index to string and string to index
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        # Threshold for add word to dictionary
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, data_path):
        self.freq_threshold = 1
        frequencies = {}
        idx = 4
        num_object = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 33, 35,
                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69,
                      71, 72, 75]
        descriptions_path = open(os.path.join(data_path), 'r')
        descriptions_file = json.load(descriptions_path)

        for i, obj_id in enumerate(descriptions_file.keys()):
            sentence_list = descriptions_file[str(num_object[i])]
            for word in self.tokenizer_eng(sentence_list['name_object']):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        self.create_vocab('/home/grasp/PycharmProjects/pythonProject/text_grasp/data/text_data/vocab.json')

    def read_vocab(self, file_name):
        """
        Load created vocabulary file and replace the 'index to string' and 'string to index' dictionary
        """
        vocab_path = open(file_name, 'r')
        vocab = json.load(vocab_path)
        new_itos = {int(key): value for key, value in vocab['itos'].items()}

        self.itos = new_itos
        self.stoi = vocab['stoi']

    def create_vocab(self, file_name):
        # create json object from dictionary
        vocab = json.dumps({'itos': self.itos,
                            'stoi': self.stoi})

        # open file for writing, "w"
        f = open(file_name, "w")

        # write json object to file
        f.write(vocab)

        # close file
        f.close()

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


if __name__ == "__main__":
    root = ''
    train_dataset = Text2GraspDataset(root,0, 10, split='train', remove_outlier=True,
                                     remove_invisible=True, num_points=20000)

    list_ = []
    len_ = train_dataset.__len__()
    print(len_/4)
    for i in range(len_):
        train_dataset.__getitem__(i)
