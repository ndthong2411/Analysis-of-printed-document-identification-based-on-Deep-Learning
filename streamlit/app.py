import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
#from resnet import resnet50, resnet101,resnet152, resnext50_32x4d, resnext101_32x8d,wide_resnet50_2,resnext101_64x4d,wide_resnet101_2
from torchvision.models import resnet50, resnet101,resnet152, resnext50_32x4d, resnext101_32x8d,wide_resnet50_2,resnext101_64x4d,wide_resnet101_2

# Local Modules
import setting


# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use the appropriate normalization values
])

# Function to classify an image
def classify_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()


# Streamlit application
def main():
    st.sidebar.title("Setting")

    # Choose pattern
    pattern_options = list(setting.MODEL_DICT.keys())
    pattern_selection = st.sidebar.selectbox("Choose a pattern", pattern_options)
    st.sidebar.write(f"Selected Pattern: {pattern_selection}")

    # Choose backbone model based on the selected pattern
    backbone_options = list(setting.MODEL_DICT[pattern_selection].keys())
    backbone_selection = st.sidebar.selectbox("Choose a backbone model", backbone_options)
    st.sidebar.write(f"Selected Backbone Model: {backbone_selection}")

    st.title("Image Classification")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify'):
            model_path = setting.MODEL_DICT[pattern_selection][backbone_selection]
            checkpoint = torch.load(model_path)
            
            if 'model_state_dict' in checkpoint:
                # Create the model based on the selected backbone model
                if backbone_selection == 'resnet50':
                    model = resnet50(pretrained=True)
                elif backbone_selection == 'resnet101':
                    model = resnet101(pretrained=True)
                elif backbone_selection == 'resnet152':
                    model = resnet152(pretrained=True)
                elif backbone_selection == 'resnext50_32x4d':
                    model = resnext50_32x4d(pretrained=True)
                elif backbone_selection == 'resnext101_32x8d':
                    model = resnext101_32x8d(pretrained=True)
                elif backbone_selection == 'resnext101_64x4d':
                    model = resnext101_64x4d(pretrained=True)
                elif backbone_selection == 'wide_resnet50_2':
                    model = wide_resnet50_2(pretrained=True)
                elif backbone_selection == 'wide_resnet101_2':
                    model = wide_resnet101_2(pretrained=True)
                else:
                    st.error(f"Backbone model '{backbone_selection}' not supported.")
                    return
               
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else:
                st.error("Error: Model state_dict not found in the checkpoint.")
                return
                
            class_index = classify_image(model, uploaded_image)
            st.write(f"Predicted Class: {class_index}")

            # Display the uploaded image instead of the home image
            # st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    else:
        # Display home image
        home_image = Image.open("E:/code/THESIS/printer_source/image/homepage.png")
        st.image(home_image, caption='Home Image', use_column_width=True)


if __name__ == '__main__':
    main()
