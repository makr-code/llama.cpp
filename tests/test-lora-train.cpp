#include "llama.h"
#include "llama-adapter.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// A real, small randomly initialized Llama model, generated solely for testing.
// No downloaded weights, tokenizer, private data or training corpus.
static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct temp_dir {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("llama-lora-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    temp_dir() { check(std::filesystem::create_directory(path), "create temp directory"); }
    ~temp_dir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    std::string file(const char * name) const { return (path / name).string(); }
};

static std::vector<uint8_t> bytes(const ggml_tensor * t) {
    std::vector<uint8_t> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, data.size());
    return data;
}

static std::vector<char> file_bytes(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    check(file.good(), "open exported file");
    return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
}

static void make_model(const std::string & path, bool quantized) {
    ggml_context_ptr ctx { ggml_init({ 2*1024*1024, nullptr, false }) };
    gguf_context_ptr meta { gguf_init_empty() };
    check(ctx && meta, "fixture allocation");
    gguf_set_val_str(meta.get(), "general.architecture", "llama");
    gguf_set_val_str(meta.get(), "tokenizer.ggml.model", "none");
    gguf_set_val_u32(meta.get(), "llama.vocab_size", 16);
    gguf_set_val_u32(meta.get(), "llama.context_length", 256);
    gguf_set_val_u32(meta.get(), "llama.embedding_length", 32);
    gguf_set_val_u32(meta.get(), "llama.block_count", 2);
    gguf_set_val_u32(meta.get(), "llama.feed_forward_length", 64);
    gguf_set_val_u32(meta.get(), "llama.attention.head_count", 2);
    gguf_set_val_u32(meta.get(), "llama.attention.head_count_kv", 1);
    gguf_set_val_u32(meta.get(), "llama.rope.dimension_count", 16);
    gguf_set_val_f32(meta.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    std::mt19937 rng(42);
    auto add = [&](const std::string & name, int64_t n0, int64_t n1, bool norm = false) {
        // Norms stay F32; every matrix is Q4_0 in the quantized fixture.
        const ggml_type type = quantized && !norm && n1 > 1 ? GGML_TYPE_Q4_0 : GGML_TYPE_F32;
        ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), type, n0, n1);
        ggml_set_name(t, name.c_str());
        std::vector<float> values(n0*n1);
        for (float & v : values) {
            v = norm ? 1.0f : (2.0f * float(rng() >> 8) / 16777216.0f - 1.0f) * 0.15f;
        }
        if (type == GGML_TYPE_F32) {
            std::copy(values.begin(), values.end(), static_cast<float *>(t->data));
        } else {
            check(ggml_quantize_chunk(type, values.data(), t->data, 0, n1, n0, nullptr) == ggml_nbytes(t),
                    "fixture quantization");
        }
        gguf_add_tensor(meta.get(), t);
    };
    add("token_embd.weight", 32, 16);
    add("output_norm.weight", 32, 1, true);
    add("output.weight", 32, 16);
    for (int layer = 0; layer < 2; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        add(prefix + "attn_norm.weight", 32, 1, true);
        add(prefix + "attn_q.weight", 32, 32);
        add(prefix + "attn_k.weight", 32, 16);
        add(prefix + "attn_v.weight", 32, 16);
        add(prefix + "attn_output.weight", 32, 32);
        add(prefix + "ffn_norm.weight", 32, 1, true);
        add(prefix + "ffn_gate.weight", 32, 64);
        add(prefix + "ffn_up.weight", 32, 64);
        add(prefix + "ffn_down.weight", 64, 32);
    }
    check(gguf_write_to_file(meta.get(), path.c_str(), false), "write fixture");
}

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

static model_ptr load(const std::string & path) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    return model_ptr(llama_model_load_from_file(path.c_str(), params), llama_model_free);
}

static context_ptr context(llama_model * model) {
    auto params = llama_context_default_params();
    params.n_ctx = params.n_batch = params.n_ubatch = 256;
    params.n_threads = params.n_threads_batch = 2;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = params.type_v = GGML_TYPE_F32;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

static std::vector<float> logits(llama_context * ctx) {
    llama_memory_clear(llama_get_memory(ctx), true);
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
            llama_batch_ext_init(ctx), llama_batch_ext_free);
    check(bool(batch), "batch allocation");
    for (int i = 0; i < 8; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch.get(), 0, i % 16);
        check(idx >= 0, "batch add token");
        const llama_pos pos = i;
        check(llama_batch_ext_set_pos(batch.get(), idx, &pos), "batch set position");
        check(llama_batch_ext_set_output_logits(batch.get(), idx, true), "batch set logits");
    }
    const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    batch.reset();
    check(rc == 0, "decode");
    const float * data = llama_get_logits(ctx);
    return { data, data + 8*16 };
}

static void near(const std::vector<float> & a, const std::vector<float> & b) {
    check(a.size() == b.size(), "logits size");
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]) && std::fabs(a[i] - b[i]) < 1e-5f,
                "logits parity");
    }
}

static bool reject(const ggml_tensor *, void *) { return false; }

// The upstream epoch callback has no userdata; borrowed, test-local state only.
static llama_adapter_lora * observed = nullptr;
static int gradient_steps = 0;
static std::map<ggml_tensor *, std::vector<float>> previous;
static constexpr float learning_rate = 0.01f;
static constexpr int epochs = 64;
static void gradients(bool train, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t,
        int64_t, int64_t, int64_t) {
    if (!train) {
        return;
    }
    for (const auto & pair : observed->ab_map) {
        for (ggml_tensor * t : { pair.second.a, pair.second.b }) {
            // Dynamic ggml_opt_eval invalidates its graph before the epoch
            // callback, so ggml_opt_grad_acc cannot be used here. With plain
            // SGD, no momentum or weight decay, delta / learning_rate is
            // exactly the applied loss gradient (up to F32 rounding).
            std::vector<float> values(ggml_nelements(t));
            ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
            auto & old = previous.at(t);
            double magnitude = 0;
            for (size_t i = 0; i < values.size(); ++i) {
                const float grad = (old[i] - values[i]) / learning_rate;
                check(std::isfinite(grad), "non-finite gradient");
                magnitude += std::fabs(grad);
            }
            // Zero B means dA == 0 at initialization, but dB must be nonzero.
            if (t == pair.second.b || gradient_steps > 0) {
                check(magnitude > 1e-12, "zero attention/MLP gradient");
            } else {
                check(magnitude == 0, "initial A gradient must be zero");
            }
            old = std::move(values);
        }
    }
    ++gradient_steps;
}

static void run(const temp_dir & dir, bool quantized) {
    const auto model_path = dir.file(quantized ? "q4.gguf" : "f32.gguf");
    make_model(model_path, quantized);
    auto model = load(model_path);
    check(bool(model), "load fixture");
    std::map<std::string, std::vector<uint8_t>> base;
    for (const auto & it : model->tensors_by_name) {
        base[it.first] = bytes(it.second);
    }
    auto base_ctx = context(model.get());
    check(bool(base_ctx), "base context");
    const auto base_logits = logits(base_ctx.get());
    base_ctx.reset();

    auto create = [&](int rank, float alpha, llama_opt_param_filter filter) {
        return llama_adapter_lora_init_trainable(model.get(), rank, alpha, 123, filter, nullptr);
    };
    const auto count = model->loras.size();
    check(!create(0, 2, llama_lora_target_filter_attn_mlp), "zero rank accepted");
    check(!create(33, 2, llama_lora_target_filter_attn_mlp), "excessive rank accepted");
    check(!create(2, 0, llama_lora_target_filter_attn_mlp), "zero alpha accepted");
    check(!create(2, std::numeric_limits<float>::infinity(), llama_lora_target_filter_attn_mlp), "infinite alpha");
    check(!create(2, std::numeric_limits<float>::quiet_NaN(), llama_lora_target_filter_attn_mlp), "NaN alpha");
    check(!create(2, 2, nullptr) && !create(2, 2, reject), "invalid filter accepted");
    check(!llama_adapter_lora_init_trainable(nullptr, 2, 2, 123, reject, nullptr), "null model");
    check(model->loras.size() == count, "failed init registered an adapter");
    auto * disposable = create(2, 2, llama_lora_target_filter_attn_mlp);
    check(disposable && model->loras.size() == count + 1, "register trainable adapter");
    llama_adapter_lora_free(disposable);
    check(model->loras.size() == count, "free must unregister trainable adapter");
    auto * adapter = create(2, 2, llama_lora_target_filter_attn_mlp);
    auto * twin = create(2, 2, llama_lora_target_filter_attn_mlp);
    check(adapter && twin && adapter->ab_map.size() == 14, "target count");
    check(!llama_lora_target_filter_attn_mlp(nullptr, nullptr), "null target");
    for (const auto & pair : adapter->ab_map) {
        check(bytes(pair.second.a) == bytes(twin->ab_map.at(pair.first).a), "seed determinism A");
        const auto b = bytes(pair.second.b);
        check(std::all_of(b.begin(), b.end(), [](uint8_t v) { return v == 0; }), "nonzero initial B");
    }
    const auto save1 = dir.file("one.gguf"), save2 = dir.file("two.gguf");
    check(llama_adapter_lora_save(adapter, model.get(), save1.c_str()), "initial save");
    check(llama_adapter_lora_save(twin, model.get(), save2.c_str()), "twin save");
    check(file_bytes(save1) == file_bytes(save2), "deterministic initial GGUF");
    check(!llama_adapter_lora_save(nullptr, model.get(), save1.c_str()), "null save adapter");
    check(!llama_adapter_lora_save(adapter, nullptr, save1.c_str()), "null save model");
    check(!llama_adapter_lora_save(adapter, model.get(), ""), "empty save path");
    check(!llama_adapter_lora_save(adapter, model.get(), dir.file("missing/file.gguf").c_str()), "I/O failure");
    auto wrong_model = load(model_path);
    check(bool(wrong_model), "load second model");
    check(!llama_adapter_lora_save(adapter, wrong_model.get(), save1.c_str()), "foreign model save");
    wrong_model.reset();

    auto ctx = context(model.get());
    float scale = 1.0f;
    check(bool(ctx) && llama_set_adapters_lora(ctx.get(), &adapter, 1, &scale) == 0, "attach");
    near(base_logits, logits(ctx.get()));
    llama_opt_params params {};
    params.param_filter = llama_opt_param_filter_all;
    auto optimizer = ggml_opt_get_default_optimizer_params(nullptr);
    optimizer.sgd.alpha = learning_rate;
    optimizer.sgd.wd = 0.0f;
    params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
    params.get_opt_pars_ud = &optimizer;
    params.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.train_lora_only = true;
    // Exercise clearing stale parameter flags (including F32 norms).
    for (const auto & it : model->tensors_by_name) {
        it.second->flags |= GGML_TENSOR_FLAG_PARAM;
    }
    llama_opt_init(ctx.get(), model.get(), params);
    for (const auto & it : model->tensors_by_name) {
        check(!(it.second->flags & GGML_TENSOR_FLAG_PARAM), "base marked trainable");
    }
    std::unique_ptr<ggml_opt_dataset, decltype(&ggml_opt_dataset_free)> dataset(
            ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, 256, 256, 1, 1), ggml_opt_dataset_free);
    std::vector<int32_t> tokens(256), labels(256, 3);
    for (size_t i = 0; i < tokens.size(); ++i) { tokens[i] = int32_t(i % 16); }
    ggml_backend_tensor_set(ggml_opt_dataset_data(dataset.get()), tokens.data(), 0, tokens.size()*sizeof(int32_t));
    ggml_backend_tensor_set(ggml_opt_dataset_labels(dataset.get()), labels.data(), 0, labels.size()*sizeof(int32_t));
    std::unique_ptr<ggml_opt_result, decltype(&ggml_opt_result_free)> result(ggml_opt_result_init(), ggml_opt_result_free);
    double initial = 0, final = 0;
    observed = adapter;
    gradient_steps = 0;
    previous.clear();
    for (const auto & pair : adapter->ab_map) {
        for (ggml_tensor * t : { pair.second.a, pair.second.b }) {
            auto & values = previous[t];
            values.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
        }
    }
    for (int epoch = 0; epoch < epochs; ++epoch) {
        ggml_opt_result_reset(result.get());
        llama_opt_epoch(ctx.get(), dataset.get(), result.get(), nullptr, 1, gradients, nullptr);
        ggml_opt_result_loss(result.get(), &final, nullptr);
        check(std::isfinite(final), "nonfinite loss");
        if (epoch == 0) { initial = final; }
    }
    observed = nullptr;
    previous.clear();
    printf("%s: all 14 A/B projection gradients checked; loss %.6f -> %.6f\n",
            quantized ? "Q4_0" : "F32", initial, final);
    check(gradient_steps == epochs && final < initial - 0.05, "loss did not decrease");
    for (const auto & it : model->tensors_by_name) {
        check(bytes(it.second) == base.at(it.first), "base weight changed");
    }
    for (const auto & pair : adapter->ab_map) {
        check(bytes(pair.second.a) != bytes(twin->ab_map.at(pair.first).a), "A unchanged");
        check(bytes(pair.second.b) != bytes(twin->ab_map.at(pair.first).b), "B unchanged");
    }
    // Destroy optimizer context before inference/export; model owns adapters.
    ctx.reset();
    check(llama_adapter_lora_save(adapter, model.get(), save1.c_str()), "trained save");
    auto * restored = llama_adapter_lora_init(model.get(), save1.c_str());
    check(restored && restored->alpha == adapter->alpha, "reload");
    for (const auto & pair : adapter->ab_map) {
        check(bytes(pair.second.a) == bytes(restored->ab_map.at(pair.first).a), "reload A");
        check(bytes(pair.second.b) == bytes(restored->ab_map.at(pair.first).b), "reload B");
    }
    check(llama_adapter_lora_save(restored, model.get(), save2.c_str()), "resave");
    check(file_bytes(save1) == file_bytes(save2), "save/load byte parity");
    auto original_ctx = context(model.get()), restored_ctx = context(model.get());
    check(bool(original_ctx) && bool(restored_ctx), "parity contexts");
    check(llama_set_adapters_lora(original_ctx.get(), &adapter, 1, &scale) == 0, "attach trained");
    check(llama_set_adapters_lora(restored_ctx.get(), &restored, 1, &scale) == 0, "attach restored");
    const auto trained_logits = logits(original_ctx.get());
    near(trained_logits, logits(restored_ctx.get()));
    check(trained_logits != base_logits, "training did not affect inference");
}

int main() {
    llama_backend_init();
    int status = 0;
    try {
        const temp_dir dir;
        run(dir, false);
        run(dir, true);
    } catch (const std::exception & err) {
        fprintf(stderr, "test-lora-train: %s\n", err.what());
        status = 1;
    }
    ggml_quantize_free();
    llama_backend_free();
    return status;
}
