from lectureRetriever import search_lecture
from contextRetriever import search_context

# RAG pipe
def rag_process(user_input) :
    lecture = search_lecture(user_input)
    combined_query = user_input + str(lecture)
    answer = search_context(combined_query)
    return answer 