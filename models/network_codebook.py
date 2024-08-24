import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(Codebook, self).__init__()
        self.codes = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, features):
        # Calculate distances between features and codebook entries
        dists = torch.cdist(features, self.codes)
        # Find the nearest code
        indices = torch.argmin(dists, dim=1)
        # Retrieve the corresponding codes
        return self.codes[indices]

class VQCodebook(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, initial_vectors=None):
        super(VQCodebook, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.embedding.weight.requires_grad = False  # 임베딩 업데이트 비활성화

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1 특징 벡터로 축소
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embedding_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # [-1, 1]로 정규화된 이미지를 얻기 위해 사용 (선택 사항)
        )

    def forward(self, patch, idx=None):
        z = self.encode_patch(patch)  # [embedding_dim]
        if idx is not None:
            print(f"Initializing codebook at index {idx}")
            for i in range(32): # batch_size: 32
                self.initialize_embedding_with_vector(self.embedding, z[i], idx * 32 + i)

        # Calculate distances between z and embedding
        dists = torch.cdist(z, self.embedding.weight)
        # Get the closest codebook entry
        encoding_indices = torch.argmin(dists, dim=1).unsqueeze(1)
        # Quantize the input using the closest codebook entry
        z_q = self.embedding(encoding_indices).view_as(z)

        # Calculate VQ Losses
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        codebook_loss = torch.mean((z_q - z.detach()) ** 2)
        print(f"codebook_loss: {codebook_loss}, commitment_loss: {commitment_loss}")

        reconstructed_patch = self.decode_vector(z_q)
        return reconstructed_patch, codebook_loss, commitment_loss, encoding_indices

    def encode_patch(self, patch):
        feature = self.encoder(patch)  # [batch_size, embedding_dim, 1, 1]
        return feature.view(feature.size(0), -1)  # [batch_size, embedding_dim]

    def decode_vector(self, vector):
        vector = vector.unsqueeze(2).unsqueeze(3)
        vector = nn.functional.interpolate(vector, scale_factor=(64, 64), mode='bilinear', align_corners=False)  # [batch_size, embedding_dim, 64, 64]
        decoded_patch = self.decoder(vector)  # [batch_size, 3, patch_size, patch_size]
        return decoded_patch  # [batch_size, patch_size, patch_size]

    def initialize_embedding_with_vectors(self, embedding_layer, initial_vectors):
        # Check if the shape matches
        if embedding_layer.weight.shape != initial_vectors.shape:
            raise ValueError(f"Shape of initial_vectorss {initial_vectors.shape} does not match "
                             f"embedding layer weight shape {embedding_layer.weight.shape}")

        # Initialize the embedding weights with the given vectors
        with torch.no_grad():
            embedding_layer.weight.copy_(initial_vectors)

    def initialize_embedding_with_vector(self, embedding_layer, vector, index):
        # Check if the input vector has the correct dimension
        if vector.shape[0] != embedding_layer.embedding_dim:
            raise ValueError(f"The input vector should have shape [{embedding_layer.embedding_dim}], "
                             f"but got shape {vector.shape}")

        # Initialize the specific embedding at the given index
        with torch.no_grad():
            embedding_layer.weight[index].copy_(vector)
