GENERATE_QUESTION_PROMPT = """
You are a teacher, you task is to generate questions for your students based on the given context.
Generation Guidelines:
1. Generate questions diverse and cover different aspects of the context.
2. Use the same language as the context.
3. Only use provided context to generate questions, do not make up any information.
<context>
{context}
</context>
Generate {target_num} questions.
Use `,` to separate each question.
Only Return the questions, no other text.
"""


ANSWER_WITH_REFERENCE_PROMPT = """
Your task is to answer the question based on the given context.
Generation Guidelines:
1. Use the same language as the context.
2. Only use provided context to generate answer, do not make up any information.
3. If the question is not related to the context, just return "No information found".
4. Must provide as much information as possible from the context to answer the question.
5. Answer must follow the following order: first cite the relevant context in the <reference> tag, then make inferences and finally give the answer in the <answer> tag.
<context>
{context}
</context>
<question>
{question}
</question>
"""