# import zipfile

# zip_path = r'/data/manan/ouput'  # Replace with the actual filename if needed

# extract_path = r'/data/manan/ouput_extracted'

# with zipfile.ZipFile(zip_path, "r") as zip_ref:
#     zip_ref.extractall(extract_path)
#     print(f"Extracted to: {extract_path}")

# import pickle
# pkl_path = r'/data/manan/ouput_extracted/ouput/data.pkl'

# with open(pkl_path, "rb") as f:
#     data = pickle.load(f)

# print("Loaded data type:", type(data))  # Check type (list, dict, etc.)
# if isinstance(data, dict):
#     print("Keys:", data.keys())  # Print keys if it's a dictionary
# elif isinstance(data, list):
#     print("First few items:", data[:5])  # Print first few items if it's a list
# import pickle

# # file_path = r"/data/manan/ouput_extracted/ouput/data/0"  # Change to "1" for the other file

# # with open(file_path, "rb") as f:
# #     try:
# #         data = pickle.load(f)
# #         print("Loaded successfully. Type:", type(data))
# #         print("Sample data:", data[:5] if isinstance(data, list) else list(data.keys()))
# #     except Exception as e:
#         # print("Error:", e)
# # with open('/data/manan/ouput', 'rb') as f:
# #     print(f.read(100))  # Read the first 100 bytes

# # import zipfile
# # import pickle

# # zip_path = "/data/manan/ouput"

# # with zipfile.ZipFile(zip_path, 'r') as zip_ref:
# #     # Open the pickle file within the ZIP
# #     with zip_ref.open('ouput/data.pkl', 'r') as f:
# #         data = pickle.load(f)

# # print(data)

# import zipfile
# import pickle
# import torch

# class CustomUnpickler(pickle.Unpickler):
#     def persistent_load(self, persid):
#         if isinstance(persid, tuple) and persid[0] == 'storage':
#             # Handle storage persistent ID
#             storage_type, class_type, key, location, size = persid
#             if storage_type == 'storage' and class_type == torch.FloatStorage:
#                 # Create a new storage with the specified size
#                 return torch.FloatStorage(size)
#         # Raise an error for unexpected persistent IDs
#         raise pickle.UnpicklingError(f"Unsupported persistent ID: {persid}")

# zip_path = "/data/manan/ouput"

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     # Open the pickle file within the ZIP
#     with zip_ref.open('ouput/data.pkl', 'r') as f:
#         unpickler = CustomUnpickler(f)
#         data = unpickler.load()

# print(data)






import torch

model_checkpoint = torch.load('/data/OpenDataLab___MovieNet/raw/modelsavepath/epoch_1.pth.tar', map_location='cpu', weights_only=False)

print(model_checkpoint)
