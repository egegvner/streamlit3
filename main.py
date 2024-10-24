from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd

# Define your neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 10)  # 10 output classes for digits 0-9

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # No activation here, apply softmax later
        return x

# Set Streamlit page configuration
st.set_page_config(
    page_title="PyTorch Model",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="expanded",
)

time.sleep(0.1)

# Load the PyTorch model
@st.cache_resource
def load_model():
    model = SimpleNN()  # Create an instance of your model
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()  # Switch to evaluation mode
    return model

model = load_model()

data = {
    'Layer': ['1', '2', '3', '4', '5'],
    'Neurons': [128, 256, 256, 256, 10]
}

# Helper functions for saving and loading comments
def load_comments():
    try:
        return pd.read_csv('comments.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['Name', 'Comment'])

def save_comment(name, comment):
    comments = load_comments()
    new_comment = pd.DataFrame({'Name': [name], 'Comment': [comment]})
    comments = pd.concat([comments, new_comment], ignore_index=True)
    comments.to_csv('comments.csv', index=False)

st.toast("Scroll for more!", icon="🔽")
st.write("# MNIST Digit Recognition")
st.write("###### Parameter Count: `1,480,458` ")

st.write('#### Draw a digit (0 - 9) below')

# Canvas for drawing the digit
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode='freedraw',
    key="canvas",
)

# Preprocess and predict the drawn image
if canvas_result.image_data is not None:
    input_numpy_array = np.array(canvas_result.image_data)
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image_gs = input_image.convert('L')

    input_image_gs_np = np.asarray(input_image_gs, dtype=np.float32)
    image_pil = Image.fromarray(input_image_gs_np)
    new_image = image_pil.resize((28, 28))  # Resizing to MNIST 28x28
    input_image_gs_np = np.array(new_image)

    tensor_image = np.expand_dims(input_image_gs_np, axis=0)
    tensor_image = np.expand_dims(tensor_image, axis=0)  # Add batch and channel dimensions

    # Normalize the image as per MNIST
    mean, std = 0.1307, 0.3081
    tensor_image = (tensor_image - mean) / std
    tensor_image = torch.tensor(tensor_image, dtype=torch.float32)

    # Make prediction using PyTorch model
    with torch.no_grad():
        predictions = model(tensor_image)
        output = torch.argmax(F.softmax(predictions, dim=1), dim=1).item()
        certainty = torch.max(F.softmax(predictions, dim=1)).item()

    # Display the prediction and certainty
    st.write(f'# Prediction: \v`{str(output)}`')
    st.write(f'##### Certainty: \v`{certainty * 100:.2f}%`')
    st.write("###### Model `v2.0.8`")
    st.divider()
    
    st.write("### Image As a Grayscale `NumPy` Array")
    st.write(input_image_gs_np)

    st.write("# Model Analysis")
    st.write("###### Since Last Update")

    st.write("##### \n")

    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric(label="Epochs", value=10, delta=9, help="One epoch refers to one complete pass through the entire training dataset.")
    col2.metric(label="Accuracy", value="97.06%", delta="0.52%", help="Total accuracy of the model which is calculated based on the test data.")
    col3.metric(label="Model Train Time", value="0.18h", delta="0.04h", help="Time required to fully train the model with specified epoch value. (in hours)", delta_color="inverse")

    st.divider()
    
    # Neurons chart
    st.write("# Number of Neurons")
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index('Layer'), x_label="Layer Number", y_label="Neurons")
    st.write("# ")

    # Softmax equation
    st.write("Layer 5 Softmax Activation Function")
    st.latex(r"softmax({z})=\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}")
    
    st.divider()

    # Footer
    st.markdown("""
    <img src="https://www.cutercounter.com/hits.php?id=hxpcokn&nd=9&style=1" border="0" alt="website counter"></a>
    """, unsafe_allow_html=True)
    st.write("###### Credits to `Ege Güvener` / `@egegvner` @2024")

    # Comments section
    st.write("# Leave a Comment")
    name = st.text_input('Name')
    comment = st.text_area('Comment')

    if st.button('Submit', type="primary"):
        if name and comment:
            save_comment(name, comment)
            st.success('Comment submitted successfully!')
        elif name == "":
            st.error("Don't you have a name?")
        elif comment == "":
            st.error("Why would you post an empty comment?")

    # Display existing comments
    st.subheader('Existing Comments')
    comments = load_comments()
    for i, row in comments.iterrows():
        st.write(f"**{row['Name']}**: {row['Comment']}")
