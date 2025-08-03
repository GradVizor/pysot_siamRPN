// main.cpp

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

class SiamTracker {
public:
    SiamTracker(const std::string& model_path);
    void init(const cv::Mat& frame, const cv::Rect& roi);
    cv::Rect track(const cv::Mat& frame);

private:
    // ONNX Runtime members
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    // Tracker state
    cv::Point target_pos;
    cv::Size target_sz;
    std::vector<float> template_features; // The crucial stored features

    // Helper methods for pre/post-processing
    cv::Mat get_subwindow(const cv::Mat& frame, cv::Point center_pos, int original_size, int out_size);
    void preprocess(const cv::Mat& image, std::vector<float>& output_tensor);
};


// --- Constructor: Load the ONNX model ---
SiamTracker::SiamTracker(const std::string& model_path) :
    env(ORT_LOGGING_LEVEL_WARNING, "SiamTracker"),
    session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {

    // --- Get input and output names from the model ---
    // These names MUST match what you found in Netron!
    input_names = {"template", "search"};
    output_names = {"cls", "loc"};
}

// --- Init: Process the first frame to get template features ---
void SiamTracker::init(const cv::Mat& frame, const cv::Rect& roi) {
    target_pos = cv::Point(roi.x + roi.width / 2, roi.y + roi.height / 2);
    target_sz = cv::Size(roi.width, roi.height);

    // 1. Crop the template patch from the initial frame
    // The size `127` is standard for many Siam-style trackers. Check your model.
    cv::Mat z_crop = get_subwindow(frame, target_pos, target_sz.width, 127);

    // 2. Preprocess the patch (BGR->RGB, normalize, etc.)
    std::vector<float> z_tensor_data;
    preprocess(z_crop, z_tensor_data);

    // 3. Run inference on the template branch
    // To do this, we need a dummy search input. A tensor of zeros works.
    std::vector<float> x_dummy_tensor_data(1 * 3 * 255 * 255, 0.0f); // Assuming 255x255 search size

    std::array<int64_t, 4> z_shape = {1, 3, 127, 127};
    std::array<int64_t, 4> x_shape = {1, 3, 255, 255};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value z_tensor = Ort::Value::CreateTensor<float>(memory_info, z_tensor_data.data(), z_tensor_data.size(), z_shape.data(), z_shape.size());
    Ort::Value x_tensor = Ort::Value::CreateTensor<float>(memory_info, x_dummy_tensor_data.data(), x_dummy_tensor_data.size(), x_shape.data(), x_shape.size());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(z_tensor));
    ort_inputs.push_back(std::move(x_tensor));

    // NOTE: This part is a simplification. Many trackers export a separate model/graph
    // for creating template features. If your `model.onnx` only outputs `cls` and `loc`,
    // you need to modify the ONNX export to also output the intermediate template features.
    // Let's assume for now the model has a third output: `template_feat`
    
    // A more realistic scenario is that PySOT runs the backbone on `z_crop` and saves the output.
    // Then on `track`, it runs the backbone on `x_crop` and feeds BOTH feature maps to the RPN head.
    // Your ONNX model likely combines these. You must understand how your specific ONNX model works.
    // We will assume for this example a simplified model where the tracker object handles the state.
    // For now, let's just copy the input tensor data as a placeholder.
    this->template_features = z_tensor_data;
    std::cout << "Tracker initialized." << std::endl;
}


// --- Track: Find the object in a new frame ---
cv::Rect SiamTracker::track(const cv::Mat& frame) {
    // 1. Crop the search region
    // The size `255` is standard. `context_amount` adds padding.
    float context_amount = 0.5;
    int wc_z = target_sz.width + context_amount * (target_sz.width + target_sz.height);
    int hc_z = target_sz.height + context_amount * (target_sz.width + target_sz.height);
    int s_z = round(sqrt(wc_z * hc_z));
    cv::Mat x_crop = get_subwindow(frame, target_pos, s_z, 255);

    // 2. Preprocess the search region
    std::vector<float> x_tensor_data;
    preprocess(x_crop, x_tensor_data);

    // 3. Prepare inputs for ONNX Runtime
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::array<int64_t, 4> z_shape = {1, 3, 127, 127}; // Using stored template
    std::array<int64_t, 4> x_shape = {1, 3, 255, 255};

    Ort::Value z_tensor = Ort::Value::CreateTensor<float>(memory_info, template_features.data(), template_features.size(), z_shape.data(), z_shape.size());
    Ort::Value x_tensor = Ort::Value::CreateTensor<float>(memory_info, x_tensor_data.data(), x_tensor_data.size(), x_shape.data(), x_shape.size());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(z_tensor));
    ort_inputs.push_back(std::move(x_tensor));

    // 4. Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names.data(), output_names.size());

    // 5. Post-process the outputs
    float* cls_scores = output_tensors[0].GetTensorMutableData<float>();
    float* bbox_preds = output_tensors[1].GetTensorMutableData<float>();

    // Find the location of the max score in the classification map
    // The score map is likely [1, 2, 25, 25]. We want the score for "foreground" (channel 1)
    int score_size = 25 * 25; // Check your model's output shape
    int score_map_channel_offset = score_size; 
    
    int best_idx = -1;
    float max_score = -1e9;
    for (int i = 0; i < score_size; ++i) {
        // This is a simplified softmax and penalty. PySOT does more (e.g., cosine window penalty)
        float score = cls_scores[i + score_map_channel_offset]; // Get score from the "object" channel
        if (score > max_score) {
            max_score = score;
            best_idx = i;
        }
    }

    int best_row = best_idx / 25;
    int best_col = best_idx % 25;

    // Get the corresponding bounding box prediction
    float dx = bbox_preds[best_idx];
    float dy = bbox_preds[best_idx + score_size];
    float dw = bbox_preds[best_idx + 2 * score_size];
    float dh = bbox_preds[best_idx + 3 * score_size];
    
    // --- THIS IS THE HARDEST PART TO REPLICATE ---
    // You need to precisely match the formula PySOT uses to convert dx,dy,dw,dh
    // into an updated bounding box. It involves anchor points, scaling, and penalties.
    // Below is a *simplified* interpretation.

    float scale_z = 127.0f / s_z; // A simplification
    float pred_x = (best_col / 24.0f) * 255 - (255 / 2.0f);
    float pred_y = (best_row / 24.0f) * 255 - (255 / 2.0f);
    
    // Update position
    target_pos.x += pred_x / scale_z;
    target_pos.y += pred_y / scale_z;
    
    // Update size with smoothing (lr)
    float lr = 0.3; // learning rate for size update
    target_sz.width = (1 - lr) * target_sz.width + lr * (target_sz.width * exp(dw));
    target_sz.height = (1-lr) * target_sz.height + lr * (target_sz.height * exp(dh));

    // Return the new bounding box
    return cv::Rect(target_pos.x - target_sz.width / 2,
                    target_pos.y - target_sz.height / 2,
                    target_sz.width,
                    target_sz.height);
}

// --- Preprocessing & Cropping Helpers ---
// This needs to EXACTLY match the Python implementation
void SiamTracker::preprocess(const cv::Mat& image, std::vector<float>& tensor_data) {
    cv::Mat float_image;
    // Pysot uses BGR format with mean subtraction.
    image.convertTo(float_image, CV_32FC3);

    // Normalize (example values, use what your model was trained with)
    cv::Scalar mean(104.0, 117.0, 123.0); // Example, check your config
    float_image -= mean;

    // Reshape from HWC to CHW format for the tensor
    int rows = float_image.rows;
    int cols = float_image.cols;
    int channels = float_image.channels();
    tensor_data.resize(rows * cols * channels);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < rows; ++h) {
            for (int w = 0; w < cols; ++w) {
                tensor_data[c * (rows * cols) + h * cols + w] = float_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

// A helper to crop a square region.
cv::Mat SiamTracker::get_subwindow(const cv::Mat& frame, cv::Point center_pos, int original_size, int out_size) {
    float c = (original_size + 1) / 2.0f;
    cv::Point from(center_pos.x - c, center_pos.y - c);
    cv::Mat patch;
    cv::getRectSubPix(frame, cv::Size(original_size, original_size), from, patch);
    cv::resize(patch, patch, cv::Size(out_size, out_size));
    return patch;
}




// --- Main function to run the tracker ---
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx> <path_to_video>" << std::endl;
        return -1;
    }
    std::string model_path = argv[1];
    std::string video_path = argv[2];

    // Load video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    // Create tracker
    SiamTracker tracker(model_path);

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        return -1;
    }

    // Select ROI on the first frame
    cv::Rect roi = cv::selectROI("Tracker", frame, false, false);

    // Initialize tracker
    tracker.init(frame, roi);

    // Main tracking loop
    while (cap.read(frame)) {
        cv::Rect bbox = tracker.track(frame);

        // Draw bounding box
        cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Tracker", frame);

        if (cv::waitKey(1) == 27) { // Exit on ESC key
            break;
        }
    }

    return 0;
}