# Should only use one documents
# [1] {<Document 1>}
useless_doc = """
<Documents>
{doc}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. q* should be related to the <Documents>, but the <Documents> can not provide any useful information for answering q*.
2. a* should be able to answer q*, ensuring that the response a* is accurate, detailed, and comprehensive.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
Please directly generate the question-answer pair (q*, a*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ...}}.
Ensure the quality of the generated (q*, a*, options*).
"""
# Should only use one documents
# [1] {<Document 1>}
single_doc_support = """
<Documents>
{doc}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. <Documents> can support q* by providing useful information or hints, but they do not contain explicit answers.
2. a* should use useful information from <Documents> to aid in answering q*, ensuring that the response is accurate,
detailed, and comprehensive.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
Please directly generate the question-answer pair (q*, a*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ...}}.
Ensure the quality of the generated (q*, a*).
"""
# Should retrieved multi docs
# [1] {<Document 1>}
# [2] {<Document 2>}
# [3] ...
multi_doc_support = """
<Documents>
{docs}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. Multiple documents within <Documents> can support q* by providing useful information or hints, but they do not contain
explicit answers.
2. a* should use useful information from <Documents> to aid in answering q*, ensuring that the response is accurate,
detailed, and comprehensive.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
Please directly generate the question-answer pair (q*, a*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ...}}.
Ensure the quality of the generated (q*, a*).
"""
# Should only use one documents
# [1] {<Document 1>}
single_doc_answer= """
<Documents>
{doc}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. Ensure that q* can be answered directly using the content of <Documents>, meaning its answer can be fully derived from
<Documents>.
2. a* should use the information from <Documents> to answer q* accurately, ensuring that the response is accurate, detailed,
and comprehensive.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
Please directly generate the question-answer pair (q*, a*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ...}}.
Ensure the quality of the generated (q*, a*).
"""
# Should retrieved multi docs
# [1] {<Document 1>}
# [2] {<Document 2>}
# [3] ...
multi_doc_answer = """
<Documents>
{docs}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. The answer to q* can be derived from multiple documents within <Documents>, involving multi-hop reasoning or the
integration of information from several documents.
2. a* should leverage the information in <Documents> to provide an accurate answer to q*, ensuring that the response is
accurate, detailed, and comprehensive.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
Please directly generate the question-answer pair (q*, a*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ...}}.
Ensure the quality of the generated (q*, a*).
"""