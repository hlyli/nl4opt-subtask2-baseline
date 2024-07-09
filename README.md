Parser and training implementations for various intermediate representations (IR) for [NL4Opt](https://nl4opt.github.io/) subtask 2.
The general problem being solved here is to automatically formulate a linear programming (LP) problem by identifying its variables, parameters, objective, and constraints from a natural language description. 
As an example problem,
```
Your client has $60,000 available to invest for a 1 year term. The money can be placed in a trust yielding a 2% return or in a savings account yielding a 3\% return. Based on your experience, you advise your client that at least 15% of the investment be placed in the trust and that at most 80% of the investment be placed in the savings account. How much should your client invest in each so as to maximize his return on investment?
```
then the goal is to use a language model to formulate it into an IR that can then be written as a standard form LP $$\displaystyle\min_{x} c^\top x \quad \text{such that} \quad a_i^\top x \le b_i, \quad i \in \{1, \dots, m\}.$$

This project aims to look at various IR formats that can help facilitate a language model in producing more accurate formulations. Our prior paper looked at XML and a basic tabular representation. This repository looks at a few more.

For more background information, see our previous papers and competition:
* [Rindra Ramamonjison, Haley Li, Timothy Yu, Shiqi He, Vishnu Rengan, Amin Banitalebi-dehkordi, Zirui Zhou, and Yong Zhang. 2022. Augmenting Operations Research with Auto-Formulation of Optimization Models From Problem Descriptions. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track, Association for Computational Linguistics, Abu Dhabi, UAE, 29–62. DOI:https://doi.org/10.18653/v1/2022.emnlp-industry.4](https://aclanthology.org/2022.emnlp-industry.4)
* https://nl4opt.github.io/

# RDF

[Branch](https://github.com/hlyli/nl4opt-subtask2-baseline/tree/RDF)


An issue with the XML approach was its complex syntax. The resource description framework (RDF) provides much simpler syntax of the form `subject -> predicate -> object`. We use [CodeT5+](https://huggingface.co/Salesforce/codet5p-220m) [1] to train a model that outputs lines of RDF. For example, the statement "Your client has $60,000 available" can be expressed as `available -> limit -> $60000`. 

LPWP accuracy: 65.

# Text-to-Table

[Branch](https://github.com/hlyli/nl4opt-subtask2-baseline/tree/T2T-DoubleSided)

Much of the downside of the tabular method proposed in prior work was that it required a language model to understand and perform algebraic expressions to manipulate the LP into the form $[\ A \mid b \ ]$. This is not yet feasible on language models, hence we propose a simpler form $[\ A_{\text{LHS}} \  \ b_{\text{LHS}} \  \mid \  A_{\text{RHS}} \  \  b_{\text{RHS}} \ ]$, which does not require algebraic operations, and algebra is performed as a post-processing step. The [Text-to-Table](https://github.com/shirley-wu/text_to_table) [2] model is used again for this.

LPWP accuracy: 71.

# References
1. Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi D.Q. Bui, Junnan Li, and Steven C. H. Hoi. 2023. CodeT5+: Open Code Large Language Models for Code Understanding and Generation. arXiv preprint (2023).
2. Xueqing Wu, Jiacheng Zhang, and Hang Li. 2021. Text-to-table: A new way of information extraction. arXiv preprint arXiv:2109.02707 (2021).