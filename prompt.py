def construct_input_prompt(title, abstract, categories_path):
    categories = ""
    with open(categories_path, mode="r") as f:
        categories = f.read().strip()
    
    prompt = "Based on the research categorization guidelines, classify the following project into the appropriate primary research priorities using the categories 1 to 12."
    prompt += f"\n\n{categories.strip()}\n\nProject Information:\n\n"
    prompt += f"### Title:\n'''\n{title.strip()}\n'''\n\n### Abstract:\n'''\n{abstract.strip()}\n'''\n\n"
    prompt += "Based on this information, identify the relevant research categories for this project. Provide clear explanation for your choices. Section your response in the following format:"
    prompt += "\n\n### Explanation: ...\n\n### Categories: ..."
    
    return prompt

title = "Neutralization of Primate Immunodeficiency Viruses"
abstract = "We will repurpose existing assays, techniques and expertise that are central to our project team's virology, structural biology, vaccine development and protein production skill-sets for HIV research, to now also work on SARS-CoV-2 during the COVID-19 pandemic emergency. These interactive research efforts will draw on our established methodologies and should represent a productive use of our existing NIH grant resources. We note that there continue to be institutional restrictions at all three performance sites on the effort that can be applied to our original goals relating to HIV-1 vaccine research and development. Those goals will be unchanged, but will be pursued at a reduced effort during the period when we also work on the new SARS-CoV-2 projects for which we have fewer institutional restrictions due to the COVID-19 pandemic."
categories_path = "categories.txt"

print(construct_input_prompt(title, abstract, categories_path))
