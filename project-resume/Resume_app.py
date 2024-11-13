import streamlit as st
from streamlit_tags import st_tags
import pdfplumber
import spacy
import re
import tempfile
import os
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher
import pandas as pd  # Import pandas for displaying data as a table


# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to add custom entities with EntityRuler
def add_custom_entities(nlp, user_patterns=None):
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")

    # Predefined patterns with lemmatization (you can also add exact matches without lemmatization)
    patterns = [
        {"label": "SKILL", "pattern": [{"LOWER": "python"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "sql"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "data"}, {"LOWER": "analysis"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "project"}, {"LOWER": "management"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "communication"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "digital"}, {"LOWER": "marketing"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "social"}, {"LOWER": "media"}, {"LOWER": "management"}]},
    ]

    # Add predefined patterns
    ruler.add_patterns(patterns)

    # Add user-defined patterns
    if user_patterns:
        st.write("Adding the following user-defined patterns:", user_patterns)
        ruler.add_patterns(user_patterns)

    return nlp


# Custom CSS for a clean theme with white background and red, black, and white accents
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;  /* White background for a clean look */
    }
    .stTitle {
        color: #FF6347;  /* Tomato red for title color */
        font-size: 36px;  /* Larger title font */
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .stText, .stButton {
        color: #FFD700;  /* Golden color for text and buttons */
    }
    .stImage {
        width: 120px;  /* Adjust logo size */
        height: auto;
    }
    .stFileUploader {
        background-color: #f0f0f0;  /* Light gray background for file upload */
    }
    .stTextInput, .stTextArea {
        background-color: #f8f8f8;
        color: #333333;
        border-radius: 8px;
        padding: 10px;
    }
    .stDownloadButton {
        background-color: #FF6347;
        color: white;
        border-radius: 8px;
        padding: 10px;
    }
    .stDataFrame {
        background-color: #f8f8f8;  /* Light gray background for data display */
        color: #333333;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add logo to the right of the title
logo_path = "C:\\Users\\ezer2\\Desktop\\CVs\\LOGO.jpg"  # Adjust this path to your logo
col1, col2 = st.columns([4, 1])  # Create two columns with a 4:1 ratio
with col1:
    st.title("Resume Parser with Machine Learning")
with col2:
    st.image(logo_path, width=120)  # Place the logo on the right

# App description and instructions
st.write("Upload a resume in PDF format to extract and analyze information.")
st.write("Optionally, you can add custom patterns for entity extraction.")


# Predefined skills
predefined_skills = [
    "Python", "SQL", "Machine Learning", "Data Analysis", "Project Management", "Communication", "Digital Marketing",
    "Social Media Management"
]

# Input for adding custom patterns as tags (user input)
custom_skills_input = st_tags(
    label='Add custom skills (comma-separated)',
    value=predefined_skills,  # Default skills (does not replace, just appends)
)

# Convert user input into JSON patterns if provided
user_patterns = []
if custom_skills_input:
    for skill in custom_skills_input:
        pattern = [{"LOWER": word.lower()} for word in skill.split()]
        user_patterns.append({"label": "SKILL", "pattern": pattern})

    try:
        nlp = add_custom_entities(nlp, user_patterns)
    except Exception as e:
        st.error(f"Invalid pattern format: {e}")


# Extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""  # Extract text or an empty string if no text found
        return text
    except Exception as e:
        st.error(f"Error opening PDF: {e}")
        return None


# Clean extracted text
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Extract entities
def extract_entities(text):
    doc = nlp(text)
    st.write("Tokens:", [token.text for token in doc])  # Debugging tokens

    entities = {
        "Name": [], "Email": [], "Phone": [],
        "Skills": set(), "Education": [], "Location": []
    }

    # Regex for Email and Phone
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zAZ0-9]{2,}'
    phone_pattern = r'\+?\d[\d -]{8,12}\d'

    # Extract emails and phones using regex
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    entities["Email"].extend(emails)
    entities["Phone"].extend(phones)

    # Check for Name using a more robust method
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["Name"].append(ent.text)
        elif ent.label_ == "SKILL":
            entities["Skills"].add(ent.text)  # Use a set to avoid duplicates
        elif ent.label_ == "DEGREE":
            entities["Education"].append(ent.text)
        elif ent.label_ == "GPE":  # General geopolitical entity for location
            entities["Location"].append(ent.text)

    # Use additional regex for Education extraction
    education_pattern = r'(Bachelor|Master|PhD|Degree|Diploma)[^.,;]*'
    education_matches = re.findall(education_pattern, text, flags=re.IGNORECASE)
    entities["Education"].extend(education_matches)

    # Extract name separately from cleaned text using regex for a better match
    name_pattern = r'(?<!\w)([A-Z][a-z]+(?: [A-Z][a-z]+)+)(?=\s*(Email|phone|Phone|Skills|Education|Location))'
    name_matches = re.findall(name_pattern, text)

    # Get only the matched names from tuples
    entities["Name"] = [match[0] for match in name_matches]

    # Remove duplicates and keep unique names
    entities["Name"] = list(set(entities["Name"]))

    # Convert skills set back to a list for consistency
    entities["Skills"] = list(entities["Skills"])

    return entities


# Extract relationships (e.g., experience details)
def extract_relationships(doc):
    matcher = Matcher(nlp.vocab)
    experience_pattern = [{"LOWER": "worked"}, {"LOWER": "at"}, {"ENT_TYPE": "ORG"}]
    matcher.add("Experience", [experience_pattern])

    relationships = []
    matches = matcher(doc)
    for _, start, end in matches:
        span = doc[start:end]
        relationships.append(span.text)

    # Use dependency parsing to extract more detailed relationships
    for token in doc:
        if token.dep_ == "dobj" and token.head.pos_ == "VERB":
            verb = token.head.text
            obj = token.text
            relationships.append(f"{verb} {obj}")

    return relationships


# Integrate data into a simple dictionary
def integrate_data(entities, relationships, skill_check):
    data = {
        "Name": ", ".join(entities["Name"]),
        "Email": ", ".join(entities["Email"]),
        "Phone": ", ".join(entities["Phone"]),
        "Skills": ", ".join(entities["Skills"]),
        "Education": ", ".join(entities["Education"]),
        "Location": ", ".join(entities["Location"]),
        "Experience": ", ".join(relationships),
        "Skill Wanted": "✓" if skill_check else "✗"  # Add skill check result here
    }
    return data


# Convert dictionary to CSV string
def dict_to_csv(data):
    header = "Name,Email,Phone,Skills,Education,Location,Experience,Skill Wanted\n"
    row = f"{data['Name']},{data['Email']},{data['Phone']},{data['Skills']},{data['Education']},{data['Location']},{data['Experience']},{data['Skill Wanted']}\n"
    return header + row

# Initialize session state if not already done
if "resume_data_list" not in st.session_state:
    st.session_state["resume_data_list"] = []  # Store multiple resumes' data

# Process PDF file if uploaded
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
skill_to_check = st.text_input("Enter a skill to check for (e.g., Machine Learning):")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_pdf.name

    try:
        with st.spinner("Extracting text and analyzing content..."):
            text = extract_text_from_pdf(temp_pdf_path)
            if text is None:
                st.error("No text extracted from the PDF. Please check the file.")
                st.stop()

            cleaned_text = clean_text(text)
            entities = extract_entities(cleaned_text)
            relationships = extract_relationships(nlp(cleaned_text))

            # Check if the specified skill is present in the extracted skills
            skill_check = skill_to_check.lower() in [skill.lower() for skill in entities["Skills"]]

        resume_data = integrate_data(entities, relationships, skill_check)

        # Store the resume data in session state (without overwriting previous data)
        st.session_state["resume_data_list"].append(resume_data)  # Append new resume data

        st.write("### Extracted Resume Data")

        # Convert resume data to DataFrame and display as a table
        resume_df = pd.DataFrame(st.session_state["resume_data_list"])  # Show all resumes' data
        st.dataframe(resume_df)  # This will display the data as a table

        # Convert resume data to CSV and provide download button
        csv_data = dict_to_csv(resume_data)
        download_key = f"download_button_{uploaded_file.name}"

        st.download_button(
            "Download Resume Data as CSV",
            csv_data,
            file_name="resume_data.csv",
            mime="text/csv",
            key=download_key  # Ensure unique key for each download button
        )

    except Exception as e:
        st.error(f"Error processing resume: {e}")

    # Clean up temporary file
    os.remove(temp_pdf_path)