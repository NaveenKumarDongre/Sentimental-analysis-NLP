import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image
# Fxn
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
    print(pos_list, neg_list, neu_list)
    result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
    return result 

def main():
    
    st.title("Sentiment Analysis NLP App")
    # st.subheader("Streamlit Projects")
    
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        image = Image.open('sa.png')
        st.image(image)

        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')
        
        # layout
        col1,col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                print(sentiment)
                st.write(sentiment)
                
                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")
                
                # Dataframe
                result_df = convert_to_df(sentiment)
                print(result_df)
                st.dataframe(result_df)

				# Visualization
                c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
                st.altair_chart(c,use_container_width=True)
                
            with col2:
                st.info("Token Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text)
                print(token_sentiments)
                st.write(token_sentiments)
                
    else:
        st.subheader("About")
        st.text("THis is a sentimental analysis app build with the help of Streamlit libaray")

if __name__ == "__main__":
    main()
            
            
			
        

    