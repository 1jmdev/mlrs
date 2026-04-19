use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{stdout, Write};
use std::path::Path;

use mlrs::darray::Array;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use tiktoken::CoreBpe;

const MODEL_PATH: &str = "/home/maty/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
const PROMPT: &str = "Once upon a time";
const N_HEADS: usize = 12;
const N_LAYERS: usize = 12;
const N_TOKENS: usize = 100;
const LAYER_NORM_EPS: f64 = 1e-5;

type WeightMap = HashMap<String, Array>;

#[derive(Debug)]
pub enum DemoError {
    Io(std::io::Error),
    SafeTensor(safetensors::SafeTensorError),
    MissingEncoding(&'static str),
    InvalidUtf8(std::string::FromUtf8Error),
    UnsupportedDtype { name: String, dtype: Dtype },
    MissingWeight(String),
    InvalidTokenId(f64),
    InvalidShape(String),
}

impl Display for DemoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::SafeTensor(error) => write!(f, "safetensors error: {error}"),
            Self::MissingEncoding(name) => write!(f, "missing tokenizer encoding: {name}"),
            Self::InvalidUtf8(error) => write!(f, "decoded token is not valid UTF-8: {error}"),
            Self::UnsupportedDtype { name, dtype } => {
                write!(f, "unsupported dtype for tensor {name}: {dtype}")
            }
            Self::MissingWeight(name) => write!(f, "missing weight tensor: {name}"),
            Self::InvalidTokenId(value) => write!(f, "invalid token id: {value}"),
            Self::InvalidShape(message) => write!(f, "invalid shape: {message}"),
        }
    }
}

impl Error for DemoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::SafeTensor(error) => Some(error),
            Self::InvalidUtf8(error) => Some(error),
            Self::MissingEncoding(_)
            | Self::UnsupportedDtype { .. }
            | Self::MissingWeight(_)
            | Self::InvalidTokenId(_)
            | Self::InvalidShape(_) => None,
        }
    }
}

impl From<std::io::Error> for DemoError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<safetensors::SafeTensorError> for DemoError {
    fn from(error: safetensors::SafeTensorError) -> Self {
        Self::SafeTensor(error)
    }
}

impl From<std::string::FromUtf8Error> for DemoError {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Self::InvalidUtf8(error)
    }
}

pub fn run() -> Result<(), DemoError> {
    let weights = load_weights(Path::new(MODEL_PATH))?;

    let encoding =
        tiktoken::get_encoding("r50k_base").ok_or(DemoError::MissingEncoding("r50k_base"))?;
    let prompt_ids = encoding.encode(PROMPT);
    let token_ids = ids_to_array(&prompt_ids)?;

    print!("{PROMPT}");
    let _generated = generate(&token_ids, &weights, N_TOKENS, encoding, N_HEADS, N_LAYERS)?;

    Ok(())
}

fn load_weights(path: &Path) -> Result<WeightMap, DemoError> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut weights = HashMap::with_capacity(tensors.len());

    for (name, tensor) in tensors.iter() {
        let array = tensor_to_array(name, &tensor)?;
        weights.insert(name.to_string(), array);
    }

    Ok(weights)
}

fn tensor_to_array(name: &str, tensor: &TensorView<'_>) -> Result<Array, DemoError> {
    let shape = tensor.shape();
    let data = match tensor.dtype() {
        Dtype::F32 => tensor
            .data()
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(bytes) as f64
            })
            .collect::<Vec<_>>(),
        Dtype::F64 => tensor
            .data()
            .chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = [
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ];
                f64::from_le_bytes(bytes)
            })
            .collect::<Vec<_>>(),
        dtype => {
            return Err(DemoError::UnsupportedDtype {
                name: name.to_string(),
                dtype,
            });
        }
    };

    Ok(Array::from_shape_vec(shape, data))
}

fn ids_to_array(ids: &[u32]) -> Result<Array, DemoError> {
    let data = ids.iter().map(|&id| id as f64).collect::<Vec<_>>();
    Ok(Array::from_shape_vec(&[1, ids.len()], data))
}

fn weight<'a>(weights: &'a WeightMap, name: &str) -> Result<&'a Array, DemoError> {
    weights
        .get(name)
        .ok_or_else(|| DemoError::MissingWeight(name.to_string()))
}

fn keepdims(reduced: Array, original_shape: &[usize], axis: usize) -> Array {
    let mut shape = Vec::with_capacity(original_shape.len());
    for (current_axis, &size) in original_shape.iter().enumerate() {
        if current_axis == axis {
            shape.push(1);
        } else {
            shape.push(size);
        }
    }
    let reshape = shape.iter().map(|&size| size as isize).collect::<Vec<_>>();
    reduced.reshape(&reshape)
}

fn layer_norm(x: &Array, weight: &Array, bias: &Array, eps: f64) -> Array {
    let axis = x.ndim() - 1;
    let mean = keepdims(x.mean_axis(axis), x.shape(), axis);
    let centered = x.sub_array(&mean);
    let variance = keepdims(centered.square().mean_axis(axis), x.shape(), axis);
    let normalized = centered.divide(&variance.add_scalar(eps).sqrt());
    weight.multiply(&normalized).add(bias)
}

fn gelu(x: &Array) -> Array {
    let cubic = x.powi(3).scale(0.044_715);
    let inner = x.add(&cubic).scale((2.0 / std::f64::consts::PI).sqrt());
    x.multiply(&inner.tanh().add_scalar(1.0)).scale(0.5)
}

fn softmax(x: &Array) -> Array {
    let axis = x.ndim() - 1;
    let max = keepdims(x.max_axis(axis), x.shape(), axis);
    let exp = x.sub_array(&max).exp();
    let sum = keepdims(exp.sum_axis(axis), exp.shape(), axis);
    exp.divide(&sum)
}

fn linear_3d(x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array, DemoError> {
    if x.ndim() != 3 || weight.ndim() != 2 {
        return Err(DemoError::InvalidShape(format!(
            "linear_3d expects [B, T, C] x [C, O], got {:?} and {:?}",
            x.shape(),
            weight.shape()
        )));
    }

    let batch = x.shape()[0];
    let steps = x.shape()[1];
    let channels = x.shape()[2];
    if weight.shape()[0] != channels {
        return Err(DemoError::InvalidShape(format!(
            "linear_3d channel mismatch: {:?} vs {:?}",
            x.shape(),
            weight.shape()
        )));
    }

    let output = weight.shape()[1];
    let flat = x.reshape(&[(batch * steps) as isize, channels as isize]);
    let mut projected = flat.matmul(weight);
    if let Some(bias) = bias {
        projected = projected.add(bias);
    }

    Ok(projected.reshape(&[batch as isize, steps as isize, output as isize]))
}

fn gather_token_embeddings(embedding: &Array, token_ids: &Array) -> Result<Array, DemoError> {
    if embedding.ndim() != 2 || token_ids.ndim() != 2 {
        return Err(DemoError::InvalidShape(format!(
            "token embedding gather expects [V, C] and [B, T], got {:?} and {:?}",
            embedding.shape(),
            token_ids.shape()
        )));
    }

    let batch = token_ids.shape()[0];
    let steps = token_ids.shape()[1];
    let channels = embedding.shape()[1];
    let vocab = embedding.shape()[0];
    let mut data = Vec::with_capacity(batch * steps * channels);

    for batch_index in 0..batch {
        for step_index in 0..steps {
            let token_id = token_ids.get(&[batch_index, step_index]);
            let token_index = float_to_index(token_id)?;
            if token_index >= vocab {
                return Err(DemoError::InvalidTokenId(token_id));
            }
            let start = token_index * channels;
            let end = start + channels;
            data.extend_from_slice(&embedding.data()[start..end]);
        }
    }

    Ok(Array::from_shape_vec(&[batch, steps, channels], data))
}

fn gather_position_embeddings(
    embedding: &Array,
    batch: usize,
    steps: usize,
) -> Result<Array, DemoError> {
    if embedding.ndim() != 2 {
        return Err(DemoError::InvalidShape(format!(
            "position embedding gather expects [P, C], got {:?}",
            embedding.shape()
        )));
    }

    let positions = (0..steps).collect::<Vec<_>>();
    let gathered = embedding.take(&positions, 0);
    let channels = embedding.shape()[1];
    let mut data = Vec::with_capacity(batch * steps * channels);
    for _ in 0..batch {
        data.extend_from_slice(gathered.data());
    }

    Ok(Array::from_shape_vec(&[batch, steps, channels], data))
}

fn batched_matmul_4d(left: &Array, right: &Array) -> Result<Array, DemoError> {
    if left.ndim() != 4 || right.ndim() != 4 {
        return Err(DemoError::InvalidShape(format!(
            "batched_matmul_4d expects rank-4 inputs, got {:?} and {:?}",
            left.shape(),
            right.shape()
        )));
    }

    let [batch, heads, left_rows, shared]: [usize; 4] = left
        .shape()
        .try_into()
        .map_err(|_| DemoError::InvalidShape(format!("invalid left rank: {:?}", left.shape())))?;
    let [right_batch, right_heads, right_shared, right_cols]: [usize; 4] = right
        .shape()
        .try_into()
        .map_err(|_| DemoError::InvalidShape(format!("invalid right rank: {:?}", right.shape())))?;

    if batch != right_batch || heads != right_heads || shared != right_shared {
        return Err(DemoError::InvalidShape(format!(
            "batched_matmul_4d shape mismatch: {:?} x {:?}",
            left.shape(),
            right.shape()
        )));
    }

    let left_block = left_rows * shared;
    let right_block = shared * right_cols;
    let mut data = Vec::with_capacity(batch * heads * left_rows * right_cols);

    for batch_index in 0..batch {
        for head_index in 0..heads {
            let left_offset = (batch_index * heads + head_index) * left_block;
            let right_offset = (batch_index * heads + head_index) * right_block;
            let left_matrix = Array::from_shape_vec(
                &[left_rows, shared],
                left.data()[left_offset..left_offset + left_block].to_vec(),
            );
            let right_matrix = Array::from_shape_vec(
                &[shared, right_cols],
                right.data()[right_offset..right_offset + right_block].to_vec(),
            );
            let product = left_matrix.matmul(&right_matrix);
            data.extend_from_slice(product.data());
        }
    }

    Ok(Array::from_shape_vec(
        &[batch, heads, left_rows, right_cols],
        data,
    ))
}

fn apply_causal_mask(attn: &Array) -> Result<Array, DemoError> {
    if attn.ndim() != 4 {
        return Err(DemoError::InvalidShape(format!(
            "causal mask expects [B, H, T, T], got {:?}",
            attn.shape()
        )));
    }

    let [batch, heads, rows, cols]: [usize; 4] = attn.shape().try_into().map_err(|_| {
        DemoError::InvalidShape(format!("invalid attention rank: {:?}", attn.shape()))
    })?;
    if rows != cols {
        return Err(DemoError::InvalidShape(format!(
            "causal mask expects square attention, got {:?}",
            attn.shape()
        )));
    }

    let mut masked = attn.copy();
    for batch_index in 0..batch {
        for head_index in 0..heads {
            for row in 0..rows {
                for col in (row + 1)..cols {
                    masked.set(&[batch_index, head_index, row, col], f64::NEG_INFINITY);
                }
            }
        }
    }

    Ok(masked)
}

fn attention(
    x: &Array,
    c_attn_w: &Array,
    c_attn_b: &Array,
    c_proj_w: &Array,
    c_proj_b: &Array,
    n_heads: usize,
) -> Result<Array, DemoError> {
    let [batch, steps, channels]: [usize; 3] = x.shape().try_into().map_err(|_| {
        DemoError::InvalidShape(format!("attention expects [B, T, C], got {:?}", x.shape()))
    })?;
    let head_dim = channels / n_heads;

    let qkv = linear_3d(x, c_attn_w, Some(c_attn_b))?;
    let q = qkv.slice_axis(2, 0, channels);
    let k = qkv.slice_axis(2, channels, 2 * channels);
    let v = qkv.slice_axis(2, 2 * channels, 3 * channels);

    let q = q
        .reshape(&[
            batch as isize,
            steps as isize,
            n_heads as isize,
            head_dim as isize,
        ])
        .permute_axes(&[0, 2, 1, 3]);
    let k = k
        .reshape(&[
            batch as isize,
            steps as isize,
            n_heads as isize,
            head_dim as isize,
        ])
        .permute_axes(&[0, 2, 1, 3]);
    let v = v
        .reshape(&[
            batch as isize,
            steps as isize,
            n_heads as isize,
            head_dim as isize,
        ])
        .permute_axes(&[0, 2, 1, 3]);

    let scores =
        batched_matmul_4d(&q, &k.permute_axes(&[0, 1, 3, 2]))?.scale((head_dim as f64).powf(-0.5));
    let attn = softmax(&apply_causal_mask(&scores)?);
    let out = batched_matmul_4d(&attn, &v)?
        .permute_axes(&[0, 2, 1, 3])
        .reshape(&[batch as isize, steps as isize, channels as isize]);

    linear_3d(&out, c_proj_w, Some(c_proj_b))
}

fn ffn(
    x: &Array,
    c_fc_w: &Array,
    c_fc_b: &Array,
    c_proj_w: &Array,
    c_proj_b: &Array,
) -> Result<Array, DemoError> {
    let hidden = linear_3d(x, c_fc_w, Some(c_fc_b))?;
    linear_3d(&gelu(&hidden), c_proj_w, Some(c_proj_b))
}

fn transformer_block(
    x: &Array,
    weights: &WeightMap,
    layer: usize,
    n_heads: usize,
) -> Result<Array, DemoError> {
    let ln1_weight = weight(weights, &format!("h.{layer}.ln_1.weight"))?;
    let ln1_bias = weight(weights, &format!("h.{layer}.ln_1.bias"))?;
    let attn_out = attention(
        &layer_norm(x, ln1_weight, ln1_bias, LAYER_NORM_EPS),
        weight(weights, &format!("h.{layer}.attn.c_attn.weight"))?,
        weight(weights, &format!("h.{layer}.attn.c_attn.bias"))?,
        weight(weights, &format!("h.{layer}.attn.c_proj.weight"))?,
        weight(weights, &format!("h.{layer}.attn.c_proj.bias"))?,
        n_heads,
    )?;
    let x = x.add(&attn_out);

    let ln2_weight = weight(weights, &format!("h.{layer}.ln_2.weight"))?;
    let ln2_bias = weight(weights, &format!("h.{layer}.ln_2.bias"))?;
    let ffn_out = ffn(
        &layer_norm(&x, ln2_weight, ln2_bias, LAYER_NORM_EPS),
        weight(weights, &format!("h.{layer}.mlp.c_fc.weight"))?,
        weight(weights, &format!("h.{layer}.mlp.c_fc.bias"))?,
        weight(weights, &format!("h.{layer}.mlp.c_proj.weight"))?,
        weight(weights, &format!("h.{layer}.mlp.c_proj.bias"))?,
    )?;

    Ok(x.add(&ffn_out))
}

fn forward(
    token_ids: &Array,
    weights: &WeightMap,
    n_heads: usize,
    n_layers: usize,
) -> Result<Array, DemoError> {
    let [batch, steps]: [usize; 2] = token_ids.shape().try_into().map_err(|_| {
        DemoError::InvalidShape(format!(
            "forward expects [B, T], got {:?}",
            token_ids.shape()
        ))
    })?;

    let token_embeddings = gather_token_embeddings(weight(weights, "wte.weight")?, token_ids)?;
    let position_embeddings =
        gather_position_embeddings(weight(weights, "wpe.weight")?, batch, steps)?;
    let mut x = token_embeddings.add(&position_embeddings);

    for layer in 0..n_layers {
        x = transformer_block(&x, weights, layer, n_heads)?;
    }

    let x = layer_norm(
        &x,
        weight(weights, "ln_f.weight")?,
        weight(weights, "ln_f.bias")?,
        LAYER_NORM_EPS,
    );

    linear_3d(&x, &weight(weights, "wte.weight")?.transpose(), None)
}

fn generate(
    prompt_ids: &Array,
    weights: &WeightMap,
    n_tokens: usize,
    encoding: &CoreBpe,
    n_heads: usize,
    n_layers: usize,
) -> Result<Array, DemoError> {
    let mut tokens = prompt_ids.copy();

    for _ in 0..n_tokens {
        let logits = forward(&tokens, weights, n_heads, n_layers)?;
        let last_index = tokens.shape()[1] - 1;
        let next_token = logits.slice_axis(1, last_index, last_index + 1).squeeze();
        let token_value = next_token.argmax() as f64;
        let token_id = float_to_u32(token_value)?;
        let decoded = encoding.decode_to_string(&[token_id])?;

        print!("{decoded}");
        stdout().flush()?;

        let next_token_array = Array::from_shape_vec(&[1, 1], vec![token_id as f64]);
        tokens = Array::concatenate(&[&tokens, &next_token_array], 1);
    }

    println!();
    Ok(tokens)
}

fn float_to_index(value: f64) -> Result<usize, DemoError> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(DemoError::InvalidTokenId(value));
    }
    Ok(value as usize)
}

fn float_to_u32(value: f64) -> Result<u32, DemoError> {
    let index = float_to_index(value)?;
    u32::try_from(index).map_err(|_| DemoError::InvalidTokenId(value))
}

fn main() {
    let _ = run();
}