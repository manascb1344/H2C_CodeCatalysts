import cloudinary
import cloudinary.uploader
import cloudinary.api

import json


import cloudinary
          
cloudinary.config( 
  cloud_name = "df4gpwc0y", 
  api_key = "568149677141638", 
  api_secret = "i-ivcnbd_49gWa5ghpAvvQgY9QY" 
)

def uploadImage():

  cloudinary.uploader.upload("/mnt/c/Users/mcb76/OneDrive/Desktop/output_graph.jpeg", public_id="output_graph", unique_filename = False, overwrite=True)
  srcURL = cloudinary.CloudinaryImage("output_graph").build_url()

  print("****2. Upload an image****\nDelivery URL: ", srcURL, "\n")
  
def getAssetInfo():
  image_info=cloudinary.api.resource("output_graph")
  print("****3. Get and use details of the image****\nUpload response:\n", json.dumps(image_info,indent=2), "\n")

  if image_info["width"]>900:
    update_resp=cloudinary.api.update("output_graph", tags = "large")
  elif image_info["width"]>500:
    update_resp=cloudinary.api.update("output_graph", tags = "medium")
  else:
    update_resp=cloudinary.api.update("output_graph", tags = "small")

  print("New tag: ", update_resp["tags"], "\n")
  



def main():
  uploadImage()
  getAssetInfo()

main()