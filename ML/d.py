import cloudinary
import cloudinary.uploader
import cloudinary.api

# Import to format the JSON responses
# ==============================
import json

# Set configuration parameter: return "https" URLs by setting secure=True  
# ==============================
# config = cloudinary.config( 
#   cloud_name = "sample", 
#   api_key = "568149677141638", 
#   api_secret = "i-ivcnbd_49gWa5ghpAvvQgY9QY@df4gpwc0y",
#   secure = True
# )

import cloudinary
          
cloudinary.config( 
  cloud_name = "df4gpwc0y", 
  api_key = "568149677141638", 
  api_secret = "i-ivcnbd_49gWa5ghpAvvQgY9QY" 
)

def uploadImage():

  # Upload the image and get its URL
  # ==============================

  # Upload the image.
  # Set the asset's public ID and allow overwriting the asset with new versions
  cloudinary.uploader.upload("/mnt/c/Users/mcb76/OneDrive/Desktop/output_graph.jpeg", public_id="output_graph", unique_filename = False, overwrite=True)

  # Build the URL for the image and save it in the variable 'srcURL'
  srcURL = cloudinary.CloudinaryImage("output_graph").build_url()

  # Log the image URL to the console. 
  # Copy this URL in a browser tab to generate the image on the fly.
  print("****2. Upload an image****\nDelivery URL: ", srcURL, "\n")
  
def getAssetInfo():

  # Get and use details of the image
  # ==============================

  # Get image details and save it in the variable 'image_info'.
  image_info=cloudinary.api.resource("output_graph")
  print("****3. Get and use details of the image****\nUpload response:\n", json.dumps(image_info,indent=2), "\n")

  # Assign tags to the uploaded image based on its width. Save the response to the update in the variable 'update_resp'.
  if image_info["width"]>900:
    update_resp=cloudinary.api.update("output_graph", tags = "large")
  elif image_info["width"]>500:
    update_resp=cloudinary.api.update("output_graph", tags = "medium")
  else:
    update_resp=cloudinary.api.update("output_graph", tags = "small")

  # Log the new tag to the console.
  print("New tag: ", update_resp["tags"], "\n")
  



def main():
  uploadImage()
  getAssetInfo()

main()