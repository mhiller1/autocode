import os
import gradio as gr

import autocode.generator as gen

def webapp(share, auth):
    def exit_app():
        os._exit(0)
    
    def load_model(model_path, custom_model_path, fp16, gpu):
        if model_path == "Custom":
            path = custom_model_path
        else:
            path = model_path
    
        if len(path) == 0:
            raise Exception("Empty model path")
    
        model = gen.Generator(model_path=path, fp16=fp16, gpu=gpu)
        return [str("Loaded model ") + path, model]
    
    def model_generate(model, input, max_length, return_sequences, beams, temperature) -> str:
        if isinstance(model, gen.Generator):
            return model.generate(input, int(max_length), int(return_sequences), int(beams), temperature)
        else:
            return str("Model not loaded")

    with gr.Blocks() as gui:
        loaded_model = gr.State({})
        gr.Button("Terminate").click(exit_app)
        gr.Markdown("AI code generator")
        status = gr.Textbox("Model not loaded", label="Status", interactive=False)
        with gr.Tab("Model loader"):
            model_selector = gr.Dropdown(["NinedayWang/PolyCoder-160M",
                                          "NinedayWang/PolyCoder-0.4B",
                                          "NinedayWang/PolyCoder-2.7B",
                                          "Salesforce/codegen-350M-mono",
                                          "Salesforce/codegen-2B-mono",
                                          "Salesforce/codegen-6B-mono",
                                          "Salesforce/codegen-16B-mono",
                                          "Salesforce/codegen-350M-multi",
                                          "Salesforce/codegen-2B-multi",
                                          "Salesforce/codegen-6B-multi",
                                          "Salesforce/codegen-16B-multi",
                                          "Custom"], value="Custom")
            custom_selector = gr.Textbox("", label="Custom model path", max_lines=1)
            gpu_checkbox = gr.Checkbox(False, label="Run on GPU")
            half_checkbox = gr.Checkbox(False, label="Run in FP16 mode")
            load_button = gr.Button("Load")
            load_button.click(load_model, inputs=[model_selector, custom_selector, half_checkbox, gpu_checkbox], outputs=[status, loaded_model])
    
        with gr.Tab("Code generator"):
            with gr.Column():
                input_text = gr.Textbox("", label="Input data")
                with gr.Row():
                    input_max_length = gr.Number(128, label="Output sequence length")
                    input_return_sequences = gr.Number(1, label="Number of output sequences")
                    input_beams = gr.Number(4, label="Beam count")
                input_temperature = gr.Slider(value=0.8, minimum=0.1, maximum=1.0, label="Model temperature")
            output_text = gr.Textbox("", label="Output data", interactive=False)
            submit_button = gr.Button("Submit")
            submit_button.click(model_generate, inputs=[loaded_model, input_text, input_max_length, input_return_sequences, input_beams, input_temperature], outputs=[output_text])
            
    gui.queue(concurrency_count=1)
    gui.launch(share=share, auth=auth)

if __name__ == '__main__':
    webapp(False, None)
 