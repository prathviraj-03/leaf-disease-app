import streamlit as st
import tensorflow as tf
import numpy as np

# Custom CSS for improved aesthetics
st.markdown(
    """
    <style>
    body {
        background-image: url("https://www.transparenttextures.com/patterns/light-paper-fibers.png");
        background-size: cover;
    }
    .title-wrapper {
        padding: 2rem;
        background-color: #4CAF50;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .title-text {
        font-size: 3rem; 
        color: #ffffff;
        font-weight: bold;
    }        
    .header-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #4CAF50;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .content {
        margin-top: 20px;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        margin-top: 10px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if app_mode == "Home":
    st.markdown('<div class="title-wrapper"><h1 class="title-text">Leaf Disease Detection</h1></div>', unsafe_allow_html=True)
    image_path = "BBG.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Plant Disease Recognition System! 🌿🔍

        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
        """)

elif app_mode == "About":
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.title("About the Project")
    
    st.markdown("""
        ### Dataset Information
        This application utilizes the **PlantVillage dataset**, a comprehensive collection of plant leaf images for training machine learning models to identify various plant diseases.

        **Key features of the dataset:**
        - **Extensive Collection:** Over 87,000 RGB images of plant leaves.
        - **Diverse Classes:** 38 categories representing diseased and healthy plants.
        - **Data Split:** 80% training, 20% validation, and a separate test set with 33 images.
        - **Image Augmentation:** Offline augmentation enhances generalization.

        The dataset is publicly available on Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

        ### Classes in the Dataset
    """)

    classes = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    st.markdown("<ul>", unsafe_allow_html=True)
    for cls in classes:
        st.markdown(f"<li>{cls}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

    st.markdown("""
        ### Developer Information

        **Developed by:** [Prathviraj J Acharya](https://prathviraj.onrender.com)

        📧 Email: [prathvirajacharya0407@gmail.com](mailto:prathvirajacharya0407@gmail.com)  
        💼 LinkedIn: [linkedin.com/in/prathviraj-j-acharya](https://www.linkedin.com/in/prathviraj-j-acharya)  
        💻 GitHub: [github.com/prathviraj-03](https://github.com/prathviraj-03)
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)


#Prediction Page
elif app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        #Predict button
        if st.button("Predict"):
            st.success("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            predicted_class = class_name[result_index] if result_index < len(class_name) else "Unknown"
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
            # Display additional information based on the predicted class
            if predicted_class == 'Apple___Apple_scab':
                st.write("Apple scab, caused by the fungus *Venturia inaequalis*, creates dark, velvety spots on leaves and fruit. This can lead to curled, prematurely falling leaves, reducing fruit yield and quality. **Prevention:** Rake and destroy fallen leaves in autumn to reduce fungal spores. Prune trees to improve air circulation, and apply preventative fungicides starting in early spring. Planting resistant apple varieties is a highly effective long-term strategy.")
            elif predicted_class == 'Apple___Black_rot':
                st.write("Black rot, from the fungus *Botryosphaeria obtusa*, shows as concentric dark brown to black lesions on leaves and fruit, and can cause cankers and fruit rot. **Prevention:** Prune out and destroy cankered limbs and infected fruit. Maintain good air circulation through proper pruning. Fungicide sprays during the growing season can provide additional protection, especially during warm, humid weather.")
            elif predicted_class == 'Apple___Cedar_apple_rust':
                st.write("Cedar apple rust, caused by the fungus *Gymnosporangium juniperi-virginianae*, creates bright orange-yellow spots on apple leaves. **Prevention:** The most effective method is to remove nearby cedar or juniper trees, which are alternate hosts for the fungus. If this is not possible, apply fungicides to apple trees from the pink-bud stage until after petal fall. Resistant apple varieties are also available.")
            elif predicted_class == 'Apple___healthy':
                st.write("A healthy apple leaf is vibrant green and free of blemishes. **Best Practices:** Ensure consistent watering, especially during dry periods. Apply a balanced fertilizer in the spring. Prune annually to remove dead wood and improve air circulation. Regularly inspect leaves for early signs of pests or diseases to address issues promptly.")
            elif predicted_class == 'Blueberry___healthy':
                st.write("Healthy blueberry leaves are dark green and firm. **Best Practices:** Blueberries thrive in acidic soil (pH 4.5-5.5). Use mulch to retain soil moisture and suppress weeds. Water regularly, providing about 1-2 inches of water per week. Prune in late winter to remove old or weak canes and encourage new growth.")
            elif predicted_class == 'Cherry_(including_sour)___Powdery_mildew':
                st.write("Powdery mildew, from the fungus *Podosphaera clandestina*, creates a white, powdery coating on leaves and fruit. **Prevention:** Ensure good air circulation by pruning trees. Avoid over-fertilizing, which encourages susceptible new growth. Apply fungicides (such as sulfur, neem oil, or potassium bicarbonate) at the first sign of disease and repeat as needed.")
            elif predicted_class == 'Cherry_(including_sour)___healthy':
                st.write("Healthy cherry leaves are glossy green and free of blemishes. **Best Practices:** Provide well-drained soil and consistent moisture. Fertilize in early spring before new growth begins. Prune annually to maintain an open structure for good air circulation and sunlight penetration, which helps prevent fungal diseases.")
            elif predicted_class == 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':
                st.write("Gray leaf spot, caused by *Cercospora zeae-maydis*, creates rectangular gray or tan lesions on leaves. **Prevention:** Practice crop rotation with non-host crops. Tillage can help bury infected residue. Choose resistant corn hybrids. Fungicides may be necessary in high-risk situations, especially during warm, humid weather.")
            elif predicted_class == 'Corn_(maize)___Common_rust_':
                st.write("Common rust, from the fungus *Puccinia sorghi*, forms reddish-brown pustules on leaves. **Prevention:** The most effective management strategy is planting resistant corn hybrids. While fungicides are available, they are often not economically justified unless the infection is severe and occurs early in the season on a susceptible hybrid.")
            elif predicted_class == 'Corn_(maize)___Northern_Leaf_Blight':
                st.write("Northern leaf blight, from *Exserohilum turcicum*, creates long, gray-green lesions on leaves. **Prevention:** Plant resistant hybrids. Practice crop rotation and tillage to reduce fungal residue. Fungicides can be effective but should be applied based on scouting and disease pressure to ensure cost-effectiveness.")
            elif predicted_class == 'Corn_(maize)___healthy':
                st.write("Healthy corn leaves are vibrant green and without blemishes. **Best Practices:** Ensure adequate nitrogen fertilization, as corn is a heavy feeder. Maintain consistent soil moisture, especially during the critical tasseling and silking stages. Monitor for pests like corn borers and earworms and treat as necessary.")
            elif predicted_class == 'Grape___Black_rot':
                st.write("Black rot, from the fungus *Guignardia bidwellii*, causes black lesions on leaves, shoots, and fruit. **Prevention:** Practice good sanitation by removing and destroying infected plant material, including mummified berries. Prune vines to improve air circulation. Apply fungicides from early spring through mid-summer, especially during wet periods.")
            elif predicted_class == 'Grape___Esca_(Black_Measles)':
                st.write("Esca (Black Measles) is a complex fungal disease causing dark spots on leaves and berries. **Prevention:** There is no cure for Esca. Management focuses on preventing infection. Avoid large pruning wounds, and if necessary, treat them with a wound sealant. Remove and destroy severely infected vines to reduce the spread of inoculum.")
            elif predicted_class == 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
                st.write("Leaf blight, from *Pseudocercospora vitis*, causes angular brown spots on leaves. **Prevention:** Rake and destroy fallen leaves to reduce fungal spores. Improve air circulation through proper pruning and vine training. Fungicide applications used for other grape diseases, like black rot, will also typically control leaf blight.")
            elif predicted_class == 'Grape___healthy':
                st.write("Healthy grape leaves are bright green and blemish-free. **Best Practices:** Grapes require full sun and well-drained soil. Prune annually during dormancy to select fruiting canes and remove old wood. Use a trellis system to support the vines and improve air circulation. Monitor for pests like Japanese beetles and grape berry moths.")
            elif predicted_class == 'Orange___Haunglongbing_(Citrus_greening)':
                st.write("Huanglongbing (HLB), or citrus greening, is a devastating bacterial disease spread by the Asian citrus psyllid. **Prevention:** There is no cure. Management relies on controlling the psyllid insect vector through insecticides. Remove and destroy infected trees immediately to prevent further spread. Plant only certified disease-free trees.")
            elif predicted_class == 'Peach___Bacterial_spot':
                st.write("Bacterial spot, from *Xanthomonas arboricola pv. pruni*, causes dark lesions on leaves and fruit. **Prevention:** Plant resistant peach varieties if available. Apply copper-based bactericides in the fall after leaf drop and in the spring before bud swell. Prune to improve air circulation. Avoid high-nitrogen fertilizers, which can increase susceptibility.")
            elif predicted_class == 'Peach___healthy':
                st.write('Healthy peach leaves are deep green and blemish-free. **Best Practices:** Peaches require full sun and well-drained soil. Prune in late winter to an open center or "vase" shape to promote air circulation and sunlight penetration. Thin fruit to prevent branches from breaking and to increase the size of remaining peaches.')
            elif predicted_class == 'Pepper,_bell___Bacterial_spot':
                st.write("Bacterial spot, from *Xanthomonas campestris pv. vesicatoria*, causes water-soaked spots on leaves and fruit. **Prevention:** Plant resistant bell pepper varieties. Avoid overhead watering to keep foliage dry. Use copper-based bactericides as a preventative measure, especially during warm, wet weather. Rotate crops and remove infected plant debris.")
            elif predicted_class == 'Pepper,_bell___healthy':
                st.write("Healthy bell pepper leaves are glossy green and blemish-free. **Best Practices:** Plant in a sunny location with well-drained soil. Use mulch to conserve moisture and prevent weeds. Fertilize with a balanced fertilizer, but avoid excessive nitrogen. Support plants with stakes or cages to prevent branches from breaking.")
            elif predicted_class == 'Potato___Early_blight':
                st.write("Early blight, from *Alternaria solani*, creates concentric brown spots on leaves. **Prevention:** Plant certified disease-free seed potatoes. Practice crop rotation. Destroy volunteer potato plants and weeds. Apply fungicides when conditions are favorable for disease development (warm and humid).")
            elif predicted_class == 'Potato___Late_blight':
                st.write("Late blight, from *Phytophthora infestans*, causes water-soaked lesions on leaves and can lead to rapid plant collapse. **Prevention:** Plant resistant varieties. Eliminate cull piles and volunteer potato plants. Time irrigation to allow foliage to dry before evening. Apply fungicides preventatively, especially during cool, wet weather.")
            elif predicted_class == 'Potato___healthy':
                st.write('Healthy potato leaves are dark green and free of blemishes. **Best Practices:** Plant in well-drained, loose soil. "Hilling" soil around the base of the plants protects tubers from sunlight and pests. Maintain consistent moisture, especially when tubers are forming. Monitor for pests like the Colorado potato beetle.')
            elif predicted_class == 'Raspberry___healthy':
                st.write("Healthy raspberry leaves are vibrant green and blemish-free. **Best Practices:** Plant in a sunny spot with well-drained soil. Prune canes after they have finished fruiting to encourage new growth and remove potential disease sources. Use a trellis to support the canes and improve air circulation.")
            elif predicted_class == 'Soybean___healthy':
                st.write("Healthy soybean leaves are uniformly dark green and show no signs of spots, lesions, or discoloration. To maintain this, ensure proper irrigation to avoid water stress, use balanced fertilizers to provide essential nutrients, and regularly monitor for pests and diseases. **Prevention:** Practice crop rotation to disrupt disease cycles, and select disease-resistant soybean varieties whenever possible. Good field hygiene, such as removing crop debris, can also prevent the spread of pathogens.")
            elif predicted_class == 'Squash___Powdery_mildew':
                st.write("Powdery mildew, from fungi like *Erysiphe cichoracearum*, creates a white, powdery coating on leaves. **Prevention:** Plant resistant varieties. Ensure proper spacing between plants to promote air circulation. Water the soil, not the leaves, to reduce humidity. Apply fungicides like neem oil or sulfur at the first sign of infection.")
            elif predicted_class == 'Strawberry___Leaf_scorch':
                st.write("Leaf scorch, from the fungus *Diplocarpon earlianum*, creates dark purple spots on leaves. **Prevention:** Renovate strawberry beds after harvest by mowing off old leaves and removing them. Mulch with straw to reduce fungal splash. Plant resistant varieties. Fungicides can be used in severe cases.")
            elif predicted_class == 'Strawberry___healthy':
                st.write("Healthy strawberry leaves are bright green and free from spots, lesions, or discoloration. Regular watering, balanced fertilization, and pest monitoring are crucial for plant health. Healthy leaves support vigorous growth and high-quality fruit production.")
            elif predicted_class == 'Tomato___Bacterial_spot':
                st.write("Bacterial spot, caused by *Xanthomonas campestris pv. vesicatoria*, results in small, water-soaked spots on leaves, stems, and fruit. The spots can enlarge, become necrotic, and merge, leading to significant damage and reduced yield. Warm, wet conditions favor its spread, and management includes copper-based sprays and resistant varieties.")
            elif predicted_class == 'Tomato___Early_blight':
                st.write("Early blight, caused by *Alternaria solani*, presents as concentric rings on older leaves, leading to defoliation and reduced fruit quality. It thrives in warm, humid conditions. Management includes crop rotation, resistant varieties, and timely fungicide applications.")
            elif predicted_class == 'Tomato___Late_blight':
                st.write("Late blight, caused by *Phytophthora infestans*, causes water-soaked lesions on leaves and stems, quickly leading to plant collapse and significant fruit rot. Cool, wet conditions favor its spread. Management strategies include using resistant varieties, ensuring proper field sanitation, and applying fungicides.")
            elif predicted_class == 'Tomato___Leaf_Mold':
                st.write("Leaf mold, caused by *Passalora fulva*, appears as yellow spots on the upper leaf surface and olive-green to gray mold on the underside. High humidity and poor ventilation favor its development. Managing the disease involves ensuring good air circulation, reducing humidity, and applying fungicides if necessary.")
            elif predicted_class == 'Tomato___Septoria_leaf_spot':
                st.write("Septoria leaf spot, caused by *Septoria lycopersici*, results in small, water-soaked spots that develop into circular lesions with dark borders and light centers. It can cause significant defoliation and reduced yields. Management includes crop rotation, removing infected plant debris, and applying fungicides.")
            elif predicted_class == 'Tomato___Spider_mites Two-spotted_spider_mite':
                st.write("Two-spotted spider mites (*Tetranychus urticae*) cause stippling and yellowing of leaves, leading to leaf drop and reduced plant vigor. They thrive in hot, dry conditions. Management includes using miticides, introducing natural predators, and maintaining adequate moisture levels.")
            elif predicted_class == 'Tomato___Target_Spot':
                st.write("Target spot, caused by *Corynespora cassiicola*, presents as dark, concentric lesions on leaves, stems, and fruit, leading to defoliation and fruit rot. Warm, humid conditions favor its spread. Effective management includes crop rotation, resistant varieties, and fungicide applications.")
            elif predicted_class == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
                st.write("Tomato yellow leaf curl virus (TYLCV) is transmitted by whiteflies, causing yellowing and curling of leaves, stunted growth, and reduced fruit production. Management focuses on controlling whitefly populations and using resistant tomato varieties.")
            elif predicted_class == 'Tomato___Tomato_mosaic_virus':
                st.write("Tomato mosaic virus (ToMV) causes mottled, discolored leaves, stunted growth, and reduced yields. It spreads through contaminated tools, hands, and plant debris. Preventative measures include using resistant varieties, sanitizing equipment, and removing infected plants.")
            elif predicted_class == 'Tomato___healthy':
                st.write("Healthy tomato leaves are vibrant green and free from spots, lesions, or discoloration. Proper watering, balanced fertilization, and pest monitoring are essential for plant health. Healthy leaves support robust growth and high-quality fruit production.")
            else:
                st.write("Additional information for this class is not available at the moment.")
    else:
        st.info("Please upload an image.")
