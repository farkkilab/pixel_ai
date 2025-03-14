from sentence_transformers import SentenceTransformer, util
from PIL import Image
import ipdb
from umap import UMAP
import glob
import os

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

# Next we compute the embeddings
# To encode an image, you can use the following code:
# from PIL import Image
# encoded_image = model.encode(Image.open(filepath))
cores_files = []
cores_path = '/data/projects/pixel_project/datasets/NKI_project_TMAs'
cores_directories = [d for d in os.listdir(cores_path) if
                     os.path.isdir(os.path.join(cores_path, d)) and d.startswith('TMA')]
for i, slide in enumerate(cores_directories):
    files_path = str(cores_path) + "/" + slide + "/Channels_all"
    cores_files.extend([os.path.join(r, fn)
                        for r, ds, fs in os.walk(files_path)
                        for fn in fs if fn.endswith('.tif')])
print("Images:", len(cores_files))
encoded_image = model.encode([Image.open(filepath) for filepath in cores_files], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

# Now we run the clustering algorithm. This function compares images aganist
# all other images and returns a list with the pairs that have the highest
# cosine similarity score
processed_images = util.paraphrase_mining_embeddings(encoded_image)
NUM_SIMILAR_IMAGES = 30

# =================
# DUPLICATES
# =================
print('Finding duplicate images...')
# Filter list for duplicates. Results are triplets (score, image_id1, image_id2) and is scorted in decreasing order
# A duplicate image will have a score of 1.00
duplicates = [image for image in processed_images if image[0] >= 1]

# Output the top X duplicate images
for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(cores_files[image_id1])
    print(cores_files[image_id2])

# =================
# NEAR DUPLICATES
# =================
print('Finding near duplicate images...')
# Use a threshold parameter to identify two images as similar. By setting the threshold lower,
# you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
# A threshold of 1.00 means the two images are exactly the same. Since we are finding near
# duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.
threshold = 0.99
near_duplicates = [image for image in processed_images if image[0] < threshold and ("TMA_42_961" in cores_files[image[1]] or "TMA_42_961" in cores_files[image[2]])]

for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
    print("Score: {:.3f}%".format(score * 100))
    print(cores_files[image_id1])
    print(cores_files[image_id2])
f = open('near_duplicates.txt', 'w')
for t in near_duplicates:
    line = ' '.join(str(x) for x in t)
    f.write(line + '\n')
f.close()

#umap_model = UMAP(n_components=2, random_state=2023, verbose=True, n_jobs=1)
#umap_encoded_images = umap_model.fit_transform(X=encoded_image)
