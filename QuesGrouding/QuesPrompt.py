# Prompt for differrent question types

class QuesPrompt:
    ExplicitReference = "There may be clear positioning keywords in the problem. Firstly, analyze and locate the keywords, such as specific text in subtitles or words spoken by someone. Then match the corresponding subtitles or audio content with the corresponding time based on this keyword. It can also assist in generating the distribution of possible video frames.\n"
    CountingOrOrdinal = "This is a counting type question. If the type of flag is the first or last, it should be searched at the beginning or end to see if there is any relevant evidence. If there is no relevant evidence, a rough range of the beginning and end should be given. If it is a problem of how many times the event occurs, the relevant content in the subtitles should be counted, and multiple related distributions should be provided to guide subsequent sampling.\n"
    Descriptive = "This is a descriptive positioning word that needs to be positioned through visual description, but can be inferred in conjunction with subtitle content. Based on the development of the subtitle event, speculate on the possible location of the content corresponding to the visual description. It can be multiple locations. If there is no clear evidence, please return a uniform sampling strategy.\n"
    Holistic = "This is a holistic issue that involves the entire video, usually determining whether the options are correct or not. The content of each option should be located in multiple segments. If there is no evidence to infer the corresponding video frame position for an option, do not fabricate it.\n" #带入验证
        
    def __init__(self):
        pass
    def get_prompt(self, question_type: str) -> str:
        # prompt = f"This question type may be {question_type} which \n"
        prompt_map = {
            "explicitreference": self.ExplicitReference,
            "counting/ordinal": self.CountingOrOrdinal,
            "descriptive": self.Descriptive,
            "holistic": self.Holistic,
        }
        
        return prompt_map.get(question_type.lower().replace(" ", ""), "Unknown problem type, with uniform distribution.")