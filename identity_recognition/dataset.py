import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


BASE_PATH = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("BODIES_AT_REST_DATA", BASE_PATH / "data_BR")
)
MAT_SHAPE = (64, 27)


def discover_subject_ids(data_root: str | Path = DEFAULT_DATA_ROOT):
    """Return subjects that contain at least one real pressure recording."""
    real_root = Path(data_root).expanduser().resolve() / "real"
    if not real_root.is_dir():
        raise FileNotFoundError(
            f"Real dataset not found at {real_root}. Pass --data_root or set "
            "BODIES_AT_REST_DATA to the data_BR directory."
        )
    subjects = [
        path.name
        for path in real_root.iterdir()
        if path.is_dir() and any(path.glob("*.p"))
    ]
    if not subjects:
        raise FileNotFoundError(f"No subject recordings (*.p) found below {real_root}")
    return sorted(subjects)


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def pressure_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.shape == (84, 47) or image.size == 84 * 47:
        image = image.reshape(84, 47)[10:74, 10:37]
    elif image.shape != MAT_SHAPE:
        if image.size != MAT_SHAPE[0] * MAT_SHAPE[1]:
            raise ValueError(
                f"Expected a {MAT_SHAPE} (or 84x47) pressure map, got {image.shape}"
            )
        image = image.reshape(MAT_SHAPE)
    return np.ascontiguousarray(np.clip(image, 0.0, 100.0) / 100.0)


class IdentityDataset(Dataset):
    """PressurePose real-data identity dataset.

    A recording file is treated as a session.  This makes it possible to train on
    ``prescribed.p`` and test on the independently collected ``p_select.p`` rather
    than leaking near-duplicate frames through a random frame split.
    """

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        sessions: tuple[str, ...] | list[str] = ("prescribed",),
        subject_ids: list[str] | None = None,
        label_to_idx: dict[str, int] | None = None,
        limit_subjects: int | None = None,
        limit_samples_per_session: int | None = None,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        discovered = discover_subject_ids(self.data_root)
        if subject_ids is None:
            subject_ids = discovered
        if limit_subjects is not None:
            subject_ids = subject_ids[:limit_subjects]
        self.subject_ids = list(subject_ids)
        self.label_to_idx = (
            dict(label_to_idx)
            if label_to_idx is not None
            else {sid: index for index, sid in enumerate(self.subject_ids)}
        )
        self.idx_to_label = {index: sid for sid, index in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        self.sessions = tuple(session.removesuffix(".p") for session in sessions)
        self.samples = []

        for subject_id in self.subject_ids:
            if subject_id not in self.label_to_idx:
                continue
            for session in self.sessions:
                path = self.data_root / "real" / subject_id / f"{session}.p"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing recording: {path}")
                recording = load_pickle(path)
                if "images" not in recording:
                    raise KeyError(f"{path} does not contain an 'images' field")
                images = recording["images"]
                count = len(images)
                if limit_samples_per_session is not None:
                    count = min(count, limit_samples_per_session)
                for image_index in range(count):
                    self.samples.append(
                        (path, image_index, self.label_to_idx[subject_id], subject_id)
                    )

        if not self.samples:
            raise ValueError("The selected subjects/sessions contain no pressure images")
        # Loading each pickle once is substantially faster than reopening it per item.
        self._recordings = {
            path: load_pickle(path)["images"] for path in {sample[0] for sample in self.samples}
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, image_index, label_idx, _ = self.samples[index]
        image = pressure_image(self._recordings[path][image_index])
        return torch.from_numpy(image[None]), label_idx
