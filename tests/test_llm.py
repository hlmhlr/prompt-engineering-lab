from promptlab.llm import DEFAULT_PARAMS, get_llm


def test_defaults_match_the_lab():
    llm = get_llm()
    assert llm.num_predict == DEFAULT_PARAMS["max_new_tokens"] == 256
    assert llm.temperature == 0.5
    assert llm.top_p == 0.2
    assert llm.top_k == 1


def test_custom_params_override_only_what_is_given():
    llm = get_llm({"max_new_tokens": 20, "temperature": 0.1})
    assert llm.num_predict == 20      # lab name mapped to Ollama's name
    assert llm.temperature == 0.1
    assert llm.top_k == 1             # untouched default
