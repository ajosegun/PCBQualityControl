import streamlit as st
from PIL import Image
import core


def main():
    st.title("PCB Hole qualification")

    # Upload image
    uploaded_file = st.file_uploader(
        "Upload a PBC image...", type=["jpg", "jpeg", "png"])
    results=[]
    if uploaded_file is not None:
        st.write(core.get_system_info())
        # Read image
        image = Image.open(uploaded_file)

        # Display image
        st.image(image, caption="Uploaded Image",
                 width=300,)  # , use_column_width=True)
        st.write(
            f"**Image Details:** Format: {image.format} / Size: {image.size} / Mode: {image.mode}")
        
        
        with st.spinner("Processing..."):
            df, image_Yolo, sam_img = core.process(image, uploaded_file.name)
            results.append([df, image_Yolo, sam_img])
        print("To display")
        
    else:
            st.info("Please upload an image file.")
        
    option = st.selectbox(
            'Choose segmentation method',
            ('YOLO', 'SAM'))
    
    try:
        if option=='YOLO':
            # Display image
            st.image(results[0][1], 
                    caption="Yolo Detection ",
                    width=300
                    )  # , use_column_width=True)
                    
        elif option=='SAM':       
            # Display image
            st.image(results[0][2], clamp=True, #channels='BGR',
                    caption="Sam Segmentation",
                    width=300
                    )  # , use_column_width=True)
        if st.button('Show Dataframe'):
            st.dataframe(results[0][0])
        else:
            st.write('No dataframe to display. Upload a PBC image and restart the process')

    except:
        st.write('Upload an imaage first please!')



if __name__ == "__main__":
    main()
