import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
# Removed numpy as it was unused
import train_script # CHANGE: Import the entire module to resolve potential path/import issues

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --- Gradio UI Logic ---

# Store results for plotting
global_results_df = pd.DataFrame(columns=['Step', 'Loss'])
global_status_message = ""

# Store CNN architecture
global_cnn_layers = []

def run_training_stream(lr, batch_size, epochs, plot_interval):
    """
    The function Gradio calls. It consumes the generator from train_script.
    Updates the plot only every 'plot_interval' batches.
    """
    global global_results_df
    global global_status_message
    
    # Reset state for a new run
    global_results_df = pd.DataFrame(columns=['Step', 'Loss'])
    global_status_message = "Starting..."
    
    # Initial yield to clear previous plot and set initial status
    yield gr.Plot(
            value=create_loss_plot(global_results_df), 
            label="Real-time Training Loss (Per Batch)"
        ), gr.Textbox(value=global_status_message)
    
    # Get the generator from the training script
    try:
        training_generator = train_script.train_model(lr, batch_size, epochs)
    except Exception as e:
        yield gr.Plot(value=create_loss_plot(global_results_df)), gr.Textbox(value=f"Error initializing model: {e}")
        return

    # Process yielded data in real-time
    for result in training_generator:
        step = result['step']
        loss = result['loss']
        status = result['status']

        global_status_message = status
        
        # Only plot steps where training is actually running (step > 0)
        if step > 0:
            # Add new data point (always collect data)
            new_row = pd.DataFrame({'Step': [step], 'Loss': [loss]})
            global_results_df = pd.concat([global_results_df, new_row], ignore_index=True)
            
            # Only update the plot every 'plot_interval' batches or on the last batch
            should_update_plot = (step % plot_interval == 0) or ("complete" in status.lower())
            
            if should_update_plot:
                # Create and yield the updated plot and status
                yield gr.Plot(
                    value=create_loss_plot(global_results_df),
                    label="Real-time Training Loss (Per Batch)"
                ), gr.Textbox(value=status)
        else:
            # Yield initial status message (step 0)
            yield gr.Plot(
                value=create_loss_plot(global_results_df),
                label="Real-time Training Loss (Per Batch)"
            ), gr.Textbox(value=status)
    
    # Final update to ensure last state is displayed
    yield gr.Plot(
        value=create_loss_plot(global_results_df),
        label="Real-time Training Loss (Per Batch)"
    ), gr.Textbox(value=global_status_message)


def create_loss_plot(df):
    """
    Generates a Matplotlib plot from the DataFrame.
    """
    if df.empty:
        # Create an empty plot when no data is available
        fig, ax = plt.subplots()
        ax.set_title("Training Loss (No data yet)")
        ax.set_xlabel("Batch Step")
        ax.set_ylabel("Loss (NLLLoss)")
        ax.set_ylim(0, 3) # Fixed Y limit for better initial visualization
        plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot data
    ax.plot(df['Step'], df['Loss'], color='orange', linestyle='-', marker='.', markersize=5, linewidth=1.5)
    
    # Styling
    ax.set_title("Real-time Training Loss", fontsize=14, fontweight='bold')
    ax.set_xlabel("Batch Step", fontsize=10)
    ax.set_ylabel("Loss (NLLLoss)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Dynamic Y-axis limits (padded)
    min_loss = df['Loss'].min()
    max_loss = df['Loss'].max()
    y_min = max(0, min_loss * 0.9)
    y_max = max_loss * 1.1 + 0.5 
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.close(fig) # Important to prevent memory leaks in Gradio
    return fig


# --- Gradio Interface Definition ---

with gr.Blocks(title="PyTorch CNN Training UI") as demo:
    gr.Markdown(
        """
        # PyTorch MNIST Trainer GUI
        Set the training parameters and run the PyTorch CNN model on a subset of the MNIST dataset.
        """
    )
    
    with gr.Tabs():
        # --- TAB 1: Training ---
        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Training Parameters")
                    
                    # Inputs
                    lr_slider = gr.Slider(
                        minimum=0.0001, 
                        maximum=0.1, 
                        value=0.001, 
                        step=0.0001, 
                        label="Learning Rate (Adam)", 
                        info="Sets the learning rate for the Adam optimizer."
                    )
                    batch_slider = gr.Slider(
                        minimum=16, 
                        maximum=256, 
                        value=64, 
                        step=16, 
                        label="Batch Size", 
                        info="Number of samples per batch (Total epochs fixed at 3)."
                    )
                    
                    epochs_slider = gr.Slider(
                        minimum=1, 
                        maximum=1000, 
                        value=3, 
                        step=1, 
                        label="Epoch number", 
                        info="Number of epochs to train"
                    )
                    
                    # New slider for controlling plot update frequency
                    plot_interval_slider = gr.Slider(
                        minimum=1, 
                        maximum=50, 
                        value=5, 
                        step=1, 
                        label="Plot Update Interval", 
                        info="Update plot every N batches (lower = more frequent updates, higher = smoother UI)"
                    )
                    
                    run_button = gr.Button("🚀 Start Training", variant="primary")

                with gr.Column(scale=3):
                    # Output Plot (Updated in real-time)
                    initial_plot = create_loss_plot(global_results_df)
                    loss_plot = gr.Plot(
                        value=initial_plot,
                        label="Real-time Training Loss (Per Batch)", 
                        every=10000.0
                        
                    )

                    # Status Output
                    status_output = gr.Textbox(
                        label="Training Status", 
                        value="Ready to start training.", 
                        lines=2, 
                        interactive=False
                    )

            # Event Binding: Click the button to start the stream
            run_button.click(
                fn=run_training_stream,
                inputs=[lr_slider, batch_slider, epochs_slider, plot_interval_slider],
                outputs=[loss_plot, status_output],
                # Use queue=True for better streaming behavior
                queue=True, 
                # Make it a generator function call
                api_name="start_training"
            )
        
        # --- TAB 2: Results Analysis ---
        with gr.Tab("Results Analysis"):
            gr.Markdown(
                """
                ## Training Results & Statistics
                View detailed analysis of your training results.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Training Summary")
                    results_display = gr.Dataframe(
                        headers=["Step", "Loss"],
                        value=global_results_df,
                        label="Training Data",
                        interactive=False
                    )
                    
                    refresh_button = gr.Button("🔄 Refresh Results", variant="secondary")
                    
                    def refresh_results():
                        global global_results_df
                        return global_results_df
                    
                    refresh_button.click(
                        fn=refresh_results,
                        outputs=results_display
                    )
                
                with gr.Column():
                    gr.Markdown("### Statistics")
                    stats_output = gr.Textbox(
                        label="Training Statistics",
                        value="No training data available yet.",
                        lines=10,
                        interactive=False
                    )
                    
                    calc_stats_button = gr.Button("📊 Calculate Statistics", variant="secondary")
                    
                    def calculate_statistics():
                        global global_results_df
                        if global_results_df.empty:
                            return "No training data available yet."
                        
                        stats = f"""
                        Total Training Steps: {len(global_results_df)}
                        
                        Loss Statistics:
                        - Initial Loss: {global_results_df['Loss'].iloc[0]:.4f}
                        - Final Loss: {global_results_df['Loss'].iloc[-1]:.4f}
                        - Minimum Loss: {global_results_df['Loss'].min():.4f}
                        - Maximum Loss: {global_results_df['Loss'].max():.4f}
                        - Average Loss: {global_results_df['Loss'].mean():.4f}
                        - Std Deviation: {global_results_df['Loss'].std():.4f}
                        
                        Improvement:
                        - Total Loss Reduction: {global_results_df['Loss'].iloc[0] - global_results_df['Loss'].iloc[-1]:.4f}
                        - Percentage Improvement: {((global_results_df['Loss'].iloc[0] - global_results_df['Loss'].iloc[-1]) / global_results_df['Loss'].iloc[0] * 100):.2f}%
                        """
                        return stats
                    
                    calc_stats_button.click(
                        fn=calculate_statistics,
                        outputs=stats_output
                    )
        
        # --- TAB 3: CNN Builder ---
        with gr.Tab("CNN Builder"):
            gr.Markdown(
                """
                ## Build Your Custom CNN Architecture
                Add layers one by one to create your custom neural network.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Add Layers")
                    
                    # Layer type selector
                    layer_type = gr.Dropdown(
                        choices=[
                            "Conv2d",
                            "MaxPool2d",
                            "AvgPool2d",
                            "ReLU",
                            "LeakyReLU",
                            "Sigmoid",
                            "Tanh",
                            "Dropout",
                            "BatchNorm2d",
                            "Linear",
                            "Flatten"
                        ],
                        value="Conv2d",
                        label="Layer Type",
                        info="Select the type of layer to add"
                    )
                    
                    # Conv2d parameters
                    with gr.Group(visible=True) as conv2d_params:
                        gr.Markdown("**Conv2d Parameters**")
                        conv_in_channels = gr.Number(value=1, label="Input Channels", precision=0)
                        conv_out_channels = gr.Number(value=10, label="Output Channels", precision=0)
                        conv_kernel_size = gr.Number(value=5, label="Kernel Size", precision=0)
                        conv_stride = gr.Number(value=1, label="Stride", precision=0)
                        conv_padding = gr.Number(value=0, label="Padding", precision=0)
                    
                    # Pooling parameters
                    with gr.Group(visible=False) as pooling_params:
                        gr.Markdown("**Pooling Parameters**")
                        pool_kernel_size = gr.Number(value=2, label="Kernel Size", precision=0)
                        pool_stride = gr.Number(value=2, label="Stride", precision=0)
                    
                    # Linear parameters
                    with gr.Group(visible=False) as linear_params:
                        gr.Markdown("**Linear Parameters**")
                        linear_in_features = gr.Number(value=320, label="Input Features", precision=0)
                        linear_out_features = gr.Number(value=10, label="Output Features", precision=0)
                    
                    # Dropout parameters
                    with gr.Group(visible=False) as dropout_params:
                        gr.Markdown("**Dropout Parameters**")
                        dropout_prob = gr.Slider(minimum=0.0, maximum=0.9, value=0.5, step=0.1, label="Dropout Probability")
                    
                    # BatchNorm parameters
                    with gr.Group(visible=False) as batchnorm_params:
                        gr.Markdown("**BatchNorm2d Parameters**")
                        bn_num_features = gr.Number(value=10, label="Number of Features", precision=0)
                    
                    # Activation parameters (LeakyReLU)
                    with gr.Group(visible=False) as leakyrelu_params:
                        gr.Markdown("**LeakyReLU Parameters**")
                        leaky_slope = gr.Slider(minimum=0.01, maximum=0.5, value=0.01, step=0.01, label="Negative Slope")
                    
                    # Update visibility based on layer type
                    def update_params_visibility(layer_type_value):
                        return {
                            conv2d_params: gr.Group(visible=(layer_type_value == "Conv2d")),
                            pooling_params: gr.Group(visible=(layer_type_value in ["MaxPool2d", "AvgPool2d"])),
                            linear_params: gr.Group(visible=(layer_type_value == "Linear")),
                            dropout_params: gr.Group(visible=(layer_type_value == "Dropout")),
                            batchnorm_params: gr.Group(visible=(layer_type_value == "BatchNorm2d")),
                            leakyrelu_params: gr.Group(visible=(layer_type_value == "LeakyReLU"))
                        }
                    
                    layer_type.change(
                        fn=update_params_visibility,
                        inputs=[layer_type],
                        outputs=[conv2d_params, pooling_params, linear_params, dropout_params, batchnorm_params, leakyrelu_params]
                    )
                    
                    # Buttons
                    with gr.Row():
                        add_layer_btn = gr.Button("➕ Add Layer", variant="primary")
                        clear_layers_btn = gr.Button("🗑️ Clear All", variant="secondary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Current Architecture")
                    
                    architecture_display = gr.Textbox(
                        label="Model Architecture",
                        value="No layers added yet. Start building your CNN!",
                        lines=20,
                        interactive=False
                    )
                    
                    # with gr.Row():
                    #     remove_last_btn = gr.Button("⬆️ Remove Last Layer", variant="secondary")
                    #     generate_code_btn = gr.Button("💾 Generate PyTorch Code", variant="primary")
                    
                    # code_output = gr.Code(
                    #     label="Generated PyTorch Model Code",
                    #     language="python",
                    #     value="# Your model code will appear here",
                    #     lines=15
                    # )
                    with gr.Row():
                        remove_last_btn = gr.Button("⬆️ Remove Last Layer", variant="secondary")
                        generate_code_btn = gr.Button("💾 Generate PyTorch Code", variant="primary")

                    # Code display
                    code_output = gr.Code(
                        label="Generated PyTorch Model Code",
                        language="python",
                        value="# Your model code will appear here",
                        lines=15
                    )

                    # NEW: Download button
                    download_btn = gr.DownloadButton(
                    "⬇️ Download custom_cnn.py"
                        )

            def download_code():
                code = generate_pytorch_code()

                filepath = os.path.join(BASE_DIR,"models", "custom_cnn.py")
                with open(filepath, "w") as f:
                    f.write(code)

                return filepath   # IMPORTANT: return only the path
            
            # Functions for CNN Builder
            def add_layer(layer_type_val, conv_in, conv_out, conv_kernel, conv_stride, conv_pad,
                         pool_kernel, pool_stride, linear_in, linear_out, dropout_p, bn_features, leaky_s):
                global global_cnn_layers
                
                layer_info = {"type": layer_type_val}
                
                if layer_type_val == "Conv2d":
                    layer_info.update({
                        "in_channels": int(conv_in),
                        "out_channels": int(conv_out),
                        "kernel_size": int(conv_kernel),
                        "stride": int(conv_stride),
                        "padding": int(conv_pad)
                    })
                elif layer_type_val in ["MaxPool2d", "AvgPool2d"]:
                    layer_info.update({
                        "kernel_size": int(pool_kernel),
                        "stride": int(pool_stride)
                    })
                elif layer_type_val == "Linear":
                    layer_info.update({
                        "in_features": int(linear_in),
                        "out_features": int(linear_out)
                    })
                elif layer_type_val == "Dropout":
                    layer_info["p"] = float(dropout_p)
                elif layer_type_val == "BatchNorm2d":
                    layer_info["num_features"] = int(bn_features)
                elif layer_type_val == "LeakyReLU":
                    layer_info["negative_slope"] = float(leaky_s)
                
                global_cnn_layers.append(layer_info)
                return format_architecture()
            
            def format_architecture():
                if not global_cnn_layers:
                    return "No layers added yet. Start building your CNN!"
                
                arch_str = "CNN Architecture:\n" + "="*50 + "\n\n"
                for idx, layer in enumerate(global_cnn_layers, 1):
                    arch_str += f"Layer {idx}: {layer['type']}\n"
                    for key, value in layer.items():
                        if key != "type":
                            arch_str += f"  - {key}: {value}\n"
                    arch_str += "\n"
                
                return arch_str
            
            def clear_layers():
                global global_cnn_layers
                global_cnn_layers = []
                return "No layers added yet. Start building your CNN!"
            
            def remove_last_layer():
                global global_cnn_layers
                if global_cnn_layers:
                    global_cnn_layers.pop()
                return format_architecture()
            
            def generate_pytorch_code():
                if not global_cnn_layers:
                    return "# No layers added yet"
                
                code = "import torch\nimport torch.nn as nn\n\n"
                code += "class CustomCNN(nn.Module):\n"
                code += "    def __init__(self):\n"
                code += "        super(CustomCNN, self).__init__()\n"
                
                for idx, layer in enumerate(global_cnn_layers, 1):
                    layer_type = layer["type"]
                    params = ", ".join([f"{k}={v}" for k, v in layer.items() if k != "type"])
                    
                    if layer_type in ["ReLU", "Sigmoid", "Tanh", "Flatten"]:
                        code += f"        self.layer{idx} = nn.{layer_type}()\n"
                    else:
                        code += f"        self.layer{idx} = nn.{layer_type}({params})\n"
                
                code += "\n    def forward(self, x):\n"
                for idx in range(1, len(global_cnn_layers) + 1):
                    code += f"        x = self.layer{idx}(x)\n"
                code += "        return x\n"
                
                return code
            
            # Event bindings for CNN Builder
            add_layer_btn.click(
                fn=add_layer,
                inputs=[
                    layer_type, conv_in_channels, conv_out_channels, conv_kernel_size, conv_stride, conv_padding,
                    pool_kernel_size, pool_stride, linear_in_features, linear_out_features,
                    dropout_prob, bn_num_features, leaky_slope
                ],
                outputs=architecture_display
            )
            
            clear_layers_btn.click(
                fn=clear_layers,
                outputs=architecture_display
            )
            
            remove_last_btn.click(
                fn=remove_last_layer,
                outputs=architecture_display
            )
            
            generate_code_btn.click(
                fn=generate_pytorch_code,
                outputs=code_output
            )
            download_btn.click(
                fn=download_code,
                inputs=None,
                outputs=download_btn
            )

            

# --- Launch the Gradio app ---
if __name__ == '__main__':
    demo.launch(share=False)