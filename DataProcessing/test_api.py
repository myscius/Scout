
from openai import OpenAI
import os

def test_openai_api_availability():
    """
    Test if the OpenAI-compatible API is available and responding correctly.
    """
    base_url = "http://automl.aiserverai.online/v1"
    api_key = "sk-HQEAuBmpuxCiNfcCEiAYZ88J217ihUK0R8ggJQidZmu09yFG"
    # base_url = "https://integrate.api.nvidia.com/v1"
    # api_key = "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"
    # base_url = "http://localhost:8000/v1"
    # api_key = "None"

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # 获取并打印当前服务端可用的模型名称
    available_models = client.models.list()
    for m in available_models.data:
        print(f"可用的模型名称: {m.id}")

    try:
    # Attempt a simple completion request to verify connectivity
        response = client.chat.completions.create(
            model="gemini-2.5-flash-lite", # deepseek-v3 Or any model name supported by your endpoint gemini-2.5-flash-lite google/gemma-4-31b-it
            messages=[
                {"role": "user", 
                 "content": '''You are a helpful assistant for video understanding.
        The user has a question about a video: "In the video, which subtitles appear at the same time as the man with black hair, dressed in grey clothes with black sleeves, on stage?
Options:
A. promisc has come to an end, in and run away countless times, i was just scared, i still
B. run away countless times, i was just scared, i still and front of our crown, like a world of souls,
C. promisc has come to an end, in and front of our crown, like a world of souls,
D. promisc has come to an end, in and captain of the godson, three three three three three three
Answer with the option letter only.
"
        
            Here are the subtitles for the video.

            NOTE that you do not need to analyze all of the subtitles. Search according to your needs to avoid too much redundant information affecting your normal work.

            <Additional subtitles, available as needed>

            [00:00:01.850-00:00:14.920] [Music] [00:00:14.930-00:00:17.660] [Applause] [00:00:17.670-00:00:26.950] [Music] [00:00:26.960-00:00:28.950] m [00:00:28.960-00:00:34.549] h I [00:00:38.270-00:00:43.190] ran away countless times, I was just scared, I still [00:00:47.590-00:00:47.600] don't know what's inside waiting at the beginning, all my tears, the day's [00:00:51.709-00:00:51.719] promise has come to an end, in [00:00:57.229-00:00:57.239] front of our crown, like a world of souls, [00:00:57.239-00:00:58.910] call me for you, [00:01:04.830-00:01:04.840] remember, finally, in the car, my future  You, the [00:01:10.429-00:01:10.439] promise of your first day in school, I won't waste it. The [00:01:13.390-00:01:13.400] captain of the godson, the [00:01:22.220-00:01:22.230] captain of the godson, three three three three three three [00:01:22.230-00:01:23.200] [Music] [00:01:23.210-00:01:28.910] [Applause] [00:01:28.920-00:01:35.389] La-la, in [00:01:38.350-00:01:38.360] the beginning, [00:01:42.030-00:01:46.789] you and I, even in [00:01:51.670-00:01:51.680] the moment of sorrow, beside your sad tears, if you break down, you and I will be happy. [00:01:57.830-00:01:57.840] But I remember you, who waited for me in my soul, my [00:02:03.709-00:02:03.719] beauty is [00:02:21.040-00:02:24.790] [music], [00:02:33.810-00:02:36.110] [applause], [00:02:42.990-00:02:43.000] my love that I made on the ke is [00:02:47.440-00:02:47.450] suddenly strong, like a light, bay, [00:02:47.450-00:02:50.589] [music]  ] [00:02:50.599-00:02:53.589] Agoyaji [00:02:58.830-00:02:58.840] Seye Seye Seye Seye Seye [00:02:58.840-00:03:00.250] Serara [00:03:00.260-00:03:09.149] [Applause] [00:03:09.159-00:03:12.159] W 
            </Additional subtitles, available as needed>

            Based on the question and the optional subtitles, please predict the temporal distribution of the relevant video segments that might contain the answer.

            There may be clear positioning keywords in the problem. Firstly, analyze and locate the keywords, such as specific text in subtitles or words spoken by someone. Then match the corresponding subtitles or audio content with the corresponding time based on this keyword. It can also assist in generating the distribution of possible video frames.

        Please analyze the context and return the result strictly in the following JSON format:
        {
            "estimated_time_range": ["string", "string"],
            "distribution": [float, float, ..., float]
        }

        Detailed Instructions:
        1. "estimated_time_range": 
           - If the answer can be explicitly located based on the subtitle timestamps, provide the time range of the corresponding subtitle segment (e.g., ["00:01:20-00:02:10"], ["00:01:20-00:02:10","00:03:30-00:04:10"]).
           - The data format is a list containing strings, which can contain one string or multiple strings. For problems of Explicit Reference type, there is usually only one string corresponding to the timestamp. For problems of the Holistic type, it is likely that each option can correspond to a timestamp string.
           - Only give a time period when the evidence is clear, otherwise it may lead to misleading. If it cannot be determined, set this value to "N/A".

        2. "distribution":
           - Divide the total video duration into 10 equal segments.
           - Provide a list of 10 probability values [p0, p1, ..., p9] corresponding to each segment. Unless absolutely certain, try not to have a complete zero probability.
           - The sum of these 10 probabilities must equal 1.0.
           - The more confident you are, the more you tend towards a unimodal distribution. Conversely, the less confident you are, the more you should tend towards a uniform or multimodal distribution.
           - Use your best judgment to assign higher probabilities to relevant segments. If you are not absolutely confident, I hope you can provide a uniform or multimodal distribution as much as possible.

        Constraint: Return ONLY the raw JSON object. Do not include Markdown blocks (```json) or any other text.'''}
            ],
            # max_tokens=300
        )
        
        print("res",response)
        # Check if we got a valid response object
        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        print(f"\nAPI Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"API connection failed: {str(e)}")

if __name__ == "__main__":
    # Allow running this script directly without pytest
    try:
        test_openai_api_availability()
        print("Test passed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")