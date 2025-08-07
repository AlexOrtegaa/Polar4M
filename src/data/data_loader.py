from torch.utils.data import Dataset

import torch
import os



class MultimodalDataset(Dataset):

    def __init__(
            self,
            data : torch.Tensor,
            list_ids: torch.Tensor,
    ) -> None:

        self._data = data
        self._list_ids = list_ids

        return

    def __len__(
            self,
    ) -> int:

        return len(self._list_ids)

    def __getitem__(
            self,
            index: int,
    ) -> torch.Tensor:
        
        sample_id = self._list_ids[index]
        return self._data[sample_id]