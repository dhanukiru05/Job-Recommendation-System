import streamlit as st
import pandas as pd
import PyPDF2
from pyresparser import ResumeParser
from sklearn.neighbors import NearestNeighbors
from src.components.job_recommender import ngrams, getNearestN, jd_df
import src.notebook.skills_extraction as skills_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to process the resume and recommend jobs
def process_resume(file_path):
    # Extract text from PDF resume
    resume_skills = skills_extraction.skills_extractor(file_path)

    # Perform job recommendation based on parsed resume data
    skills = []
    skills.append(' '.join(word for word in resume_skills))

    # Feature Engineering:
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(skills)

    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    jd_test = jd_df['Processed_JD'].values.astype('U')

    distances, indices = getNearestN(jd_test)
    matches = []

    for i, j in enumerate(indices):
        dist = round(distances[i][0], 2)
        matches.append([dist])

    matches = pd.DataFrame(matches, columns=['Match confidence'])

    # Following recommends Top 5 Jobs based on candidate resume:
    jd_df['match'] = matches['Match confidence']

    return jd_df.head(5).sort_values('match')


# Streamlit app
def main():
    # Add a creative header with subheading and description
    st.title("🚀 Job Recommendation App")
    st.subheader("Find Your Dream Job with a Few Clicks!")
    st.markdown("""
        Welcome to the *AI-powered Job Recommendation System*. 
        Upload your resume in PDF format, and we will match it to the best job openings based on your skills. 
        It's fast, simple, and tailored to help you land your next opportunity!
    """)

    # File uploader with a drag-and-drop feature
    uploaded_file = st.file_uploader("📄 Drag and Drop or Browse your resume (PDF only)", type=['pdf'])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Progress indicator
        with st.spinner('Processing your resume... ⏳'):
            file_path = uploaded_file.name
            df_jobs = process_resume(file_path)

        # Display success message and job recommendations
        st.success("🎉 Resume processed successfully! Here are your top job matches:")

        # Display recommended jobs as DataFrame with improved styling
        st.write("#### Your Top 5 Job Recommendations:")
        st.dataframe(df_jobs[['Job Title', 'Company Name', 'Location', 'Industry', 'Sector', 'Average Salary']])

        # Show additional insights with a bar chart of match confidence
        st.write("#### Match Confidence Levels:")
        st.bar_chart(df_jobs[['Job Title', 'match']].set_index('Job Title'))

        # Add an option to download the results as a CSV
        st.write("#### Download the job recommendations:")
        csv = df_jobs.to_csv(index=False)
        st.download_button(label="Download as CSV", data=csv, file_name='job_recommendations.csv', mime='text/csv')

    else:
        # Display an encouraging message if no file is uploaded yet
        st.info("📂 Please upload your resume to get job recommendations.")

# Run the Streamlit app
if __name__ == '__main__':
    main()