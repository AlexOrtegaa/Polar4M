from torch.utils.data import Dataset

import torch



class ModalityDataset(
    Dataset
):
    """Modality based PyTorch Dataset.

    This dataset is designed to handle modality-specific data by storing the data
    in a tensor and using a list of indices to access specific samples.

    Attributes:
        _data:
            A tensor containing the modality data. The shape should be (num_samples, num_features).
        _list_ids:
            A tensor containing the indices of the samples in the dataset. The shape should be (num_sub_samples,).
    """
    def __init__(
            self,
            data : torch.Tensor,
            list_ids: torch.Tensor,
    ) -> None:
        """Initialize the dataset with data and list of sample indices.

        This constructor initializes the dataset with a tensor containing the modality data
        and a tensor containing the indices of the samples in the dataset.

        Args:
            data:
                A tensor containing the modality data. The shape should be (len_dataset, num_features).
            list_ids:
                A tensor containing the indices of the samples in the dataset. The shape should be (num_samples,).
                Note len(list_ids) could be less than len(data), which means that not all samples in the data are used.
        """

        self._data = data
        self._list_ids = list_ids

        return

    def __len__(
            self,
    ) -> int:
        """Return the number of samples in the dataset.

        This method returns the number of samples in the dataset, which is determined by the length
        of the list of sample indices.

        Returns:
            The number of samples in the dataset.
        """

        return len(self._list_ids)

    def __getitem__(
            self,
            index: int,
    ) -> torch.Tensor:
        """Get a sample from the dataset by index.

        This method retrieves a sample from the dataset based on the provided index.

        Args:
            index:
                The index of the sample to retrieve from the dataset.

        Returns:
            A tensor representing the sample at the specified index.
        """
        
        sample_id = self._list_ids[index]
        return self._data[sample_id]