import streamlit as st
from PIL import Image
import core


def main():
    st.title("Upload the PCB Image")

    # Upload image
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)

        # Display image
        st.image(image, caption="Uploaded Image",
                 width=300,)  # , use_column_width=True)
        st.write(
            f"**Image Details:** Format: {image.format} / Size: {image.size} / Mode: {image.mode}")

        with st.spinner("Processing..."):
            df, image_Yolo, sam_img = core.process(image, uploaded_file.name)
        print("To display")
        
        # Display image
        st.image(image_Yolo, 
                  caption="Yolo Detection ",
                 width=300
                 )  # , use_column_width=True)
                 
                 
        # Display image
        st.image(sam_img, clamp=True, #channels='BGR',
                  caption="Sam Segmentation",
                 width=300
                 )  # , use_column_width=True)

        st.dataframe(df)
    else:
        st.info("Please upload an image file.")


if __name__ == "__main__":
    main()
