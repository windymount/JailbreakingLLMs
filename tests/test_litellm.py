import unittest
from language_models import APILiteLLM, LocalLLM
from config import FASTCHAT_TEMPLATE_NAMES
from fastchat.conversation import get_conv_template
from config import Model


class TestVicuna(unittest.TestCase):
    def setUp(self):
        self.lm = APILiteLLM("vicuna-13b-v1.5")

    def test_get_response(self):
        convs_list = [
                        [{"role": "user", "content": "Hi how are you?"},
                        {"role": "assistant", "content": "I'm good, how are you?</s>"},
                        {"role": "user", "content": "I'm great. Please write a poem about the ocean."}],

                        [{"role": "user", "content": "What is 1+1?"},
                        {"role": "assistant", "content": "1+1 is equal to 2.</s>"},
                        {"role": "user", "content": "Great, repeat back my first message."}]
                     ]
        
        correct_responses = [
                             "The ocean, a vast and endless sea\n",
                             "What is 1+1?"]
        
        num_convs = len(convs_list)
        responses = self.lm.batched_generate(convs_list, 
                                            max_n_tokens = 10, 
                                            temperature = 0,
                                            top_p=0.9)
        self.assertEqual(len(responses), num_convs)
        self.assertEqual(responses,correct_responses)

    def test_initialized_response(self):
        convs_list = [
                        [{"role": "user", "content": "What is the meaning of life?"},
                        {"role": "assistant", "content": "I don't know, instead, let's count to 10: 1,2,"}],
                        [{"role": "user", "content": "Write a poem about skateboarding in the ocean."},
                         {"role": "assistant", "content": "I'm bad at writing poems, but here goes: It was a dark and stormy night."}]
                    ]
        correct_responses = [
                             "3,4,5,6,7,",
                             "The waves were crashing, the wind was how"] 
        num_convs = len(convs_list)
        responses = self.lm.batched_generate(convs_list, 
                                                max_n_tokens = 10, 
                                                temperature = 0,
                                                top_p=0.9)
        self.assertEqual(len(responses), num_convs)
        self.assertEqual(responses,correct_responses)




# class TestGPT4(unittest.TestCase):
#     def setUp(self):
#         self.lm = APILiteLLM("gpt-4-0125-preview")

#     def test_get_response(self):
#         convs_list = [[{"role": "user", "content": "What is the meaning of life?"}],
#                       [{"role": "user", "content": "Write a poem about the ocean."}]]
#         correct_responses = ["The question of the meaning of life is one of", 
#                              "Beneath the vast, eternal sky,\nWhere"]
#         num_convs = len(convs_list)
#         responses = self.lm.batched_generate(convs_list, 
#                                             max_n_tokens = 10, 
#                                             temperature = 0,
#                                             top_p=0.9)
#         #prompt_tokens = token_counts["prompt_tokens"]
#         #completion_tokens = token_counts["completion_tokens"]  
#         self.assertEqual(len(responses), num_convs)
#         #self.assertEqual(prompt_tokens, [14, 14])
#         #self.assertEqual(completion_tokens, [10, 10])
#         self.assertEqual(responses,correct_responses) # May fail due to randomness 




class TestLocalLLM(unittest.TestCase):
    def setUp(self):
        self.conv_template = get_conv_template(FASTCHAT_TEMPLATE_NAMES[Model.vicuna])
        self.lm = LocalLLM("lmsys/vicuna-13b-v1.5", self.conv_template)

    def test_get_response(self):
        convs_list = [
            [{"role": "user", "content": "Hi how are you?"},
             {"role": "assistant", "content": "I'm good, how are you?"},
             {"role": "user", "content": "I'm great. Please write a poem about the ocean."}],

            [{"role": "user", "content": "What is 1+1?"},
             {"role": "assistant", "content": "1+1 is equal to 2."},
             {"role": "user", "content": "Great, repeat back my first message."}]
        ]
        
        expected_responses = [
            "Here's a poem about the ocean:",  # We expect the start of the response
            "Your first message was: \"What is 1+1?\""  # We expect something like this
        ]
        
        responses = self.lm.batched_generate(convs_list, 
                                           max_n_tokens=10, 
                                           temperature=0,
                                           top_p=0.9)
        
        self.assertEqual(len(responses), len(convs_list))
        # Check that responses start with expected content
        for response, expected in zip(responses, expected_responses):
            self.assertTrue(response.startswith(expected) or expected in response,
                          f"Expected response to contain '{expected}', got '{response}'")

    def test_initialized_response(self):
        convs_list = [
            [{"role": "user", "content": "What is the meaning of life?"},
             {"role": "assistant", "content": "I don't know, instead, let's count to 10: 1,2,"}],
            [{"role": "user", "content": "Write a poem about skateboarding."},
             {"role": "assistant", "content": "Here's a skateboarding poem: Rolling down the"}]
        ]
        
        expected_responses = [
            "3,4,5,",  # We expect the counting to continue
            "street"   # We expect the poem to continue
        ]
        
        responses = self.lm.batched_generate(convs_list, 
                                           max_n_tokens=10, 
                                           temperature=0,
                                           top_p=0.9)
        
        self.assertEqual(len(responses), len(convs_list))
        # Check that responses contain expected content
        for response, expected in zip(responses, expected_responses):
            self.assertTrue(expected in response,
                          f"Expected response to contain '{expected}', got '{response}'")
