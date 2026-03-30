# Vortex File Format Architecture

## 1. 文件物理布局

一个 Vortex 文件在磁盘上的结构：

```
┌───────────────────────────────────────────────┐
│  Magic Bytes (4 bytes)                        │  ← 文件标识
├───────────────────────────────────────────────┤
│                                               │
│  Segment 0: [data buffers][flatbuffer][u32]   │
│  Segment 1: [data buffers][flatbuffer][u32]   │
│  Segment 2: ...                               │
│  ...                                          │  ← 数据区：所有 segment 顺序排列
│  Segment N: ...                               │
│                                               │
├───────────────────────────────────────────────┤
│  Per-Column Statistics Buffers (optional)      │  ← 文件级统计信息
├───────────────────────────────────────────────┤
│  Schema / DType FlatBuffer (optional)         │  ← 根 schema
├───────────────────────────────────────────────┤
│  Layout FlatBuffer                            │  ← layout 树（核心元数据）
├───────────────────────────────────────────────┤
│  Footer FlatBuffer                            │  ← segment 映射表 + 编码注册表
├───────────────────────────────────────────────┤
│  Postscript FlatBuffer                        │  ← 各区域的 offset/length
├───────────────────────────────────────────────┤
│  EOF Marker (8 bytes)                         │
│  [2B version][2B postscript_len][4B magic]    │  ← 读取入口
└───────────────────────────────────────────────┘
```

**读取入口**：从文件尾部读 8 字节 EOF marker → 拿到 postscript 长度 → 读 postscript → 拿到各区域 offset → 按需读取。默认实现一次读尾部 1MB 以覆盖 postscript + footer。

## 2. 核心概念

### 2.1 逻辑概念

| 概念 | 说明 |
|------|------|
| **Chunk** | 内存中流动的一个 array。写入管线中 strategy 之间传递的数据单元 |
| **Layout 树** | 描述数据逻辑组织结构的递归树。存储在 footer 的 Layout FlatBuffer 中 |
| **LayoutStrategy** | 写入时的流转换器，决定如何组织数据。多个 strategy 组合成管线 |
| **LayoutReader** | 读取时根据 layout 树节点分发读取请求 |
| **ScanBuilder** | 用户侧读取 API，配置 projection / filter / row range 等 |

### 2.2 物理概念

| 概念 | 说明 |
|------|------|
| **Segment** | 文件中一段连续字节。Vortex 的**最小 IO 单位**。一个 segment 包含一个序列化后的 array |
| **SegmentId** | segment 的 u32 标识符（0, 1, 2, ...） |
| **SegmentSpec** | segment 的物理位置：`{ offset: u64, length: u32, alignment: Alignment }` |
| **Footer** | 文件尾部元数据，包含 layout 树 + segment 映射表 + 编码注册表 + 统计信息 |

### 2.3 逻辑与物理的映射

```
Layout 树 (逻辑)                    Segment 映射表 (物理)
┌─────────────────┐                ┌──────────────────────────────┐
│ ChunkedLayout   │                │ SegmentSpec[0]: offset=4, len=8192  │
│  ├─ StructLayout│                │ SegmentSpec[1]: offset=8196, len=4096│
│  │  ├─ Chunked  │                │ SegmentSpec[2]: offset=12292, len=...│
│  │  │  └─ Flat ─┼─ segment_id=0 ──→  ...                              │
│  │  └─ Chunked  │                └──────────────────────────────┘
│  │     └─ Flat ─┼─ segment_id=1
│  └─ StructLayout│
│     └─ ...      │
└─────────────────┘
```

每个 FlatLayout（叶节点）持有一个 `segment_id`，通过 footer 的 segment 映射表定位到文件中的具体字节范围。

## 3. Segment 内部结构

一个 segment 是一个完整序列化的 array，内部布局：

```
┌─────────────────────────────────────────────────────────────┐
│ [padding] [data buffer 0] [padding] [data buffer 1] ...     │  ← 实际数据
│ [FlatBuffer: ArrayNode 编码描述树]                            │  ← 编码元数据
│ [u32: flatbuffer 长度]                                       │  ← 尾部长度标记
└─────────────────────────────────────────────────────────────┘
```

**ArrayNode 编码描述树**（FlatBuffer 格式）：

```
ArrayNode {
    encoding: u16,              // 编码类型 ID（BtrBlocks, Primitive, Dict...）
    metadata: [u8],             // 编码特有的元数据
    buffers:  [buffer_index],   // 引用哪些 data buffer
    children: [ArrayNode],      // 子数组（递归）
    stats:    StatsSet,         // 统计信息（min, max, null_count 等）
}
```

**读取一个 segment 的过程**：
1. 从尾部读 4 字节 → 得到 flatbuffer 长度
2. 读 flatbuffer → 解析 ArrayNode → 知道编码方式 + 每个 data buffer 的 offset/长度/对齐
3. 按描述切出各个 data buffer
4. 用 encoding + metadata + buffers 反序列化回 array 对象

**关键约束**：segment 是最小 IO 单位。`SegmentSource` 接口只支持按 `SegmentId` 读取整个 segment，不支持 range read：

```rust
pub trait SegmentSource: 'static + Send + Sync {
    fn request(&self, id: SegmentId) -> SegmentFuture;
}
pub type SegmentFuture = BoxFuture<'static, VortexResult<ByteBuffer>>;
```

## 4. Layout 类型

### 4.1 FlatLayout（叶节点）

最底层的 layout，1:1 对应一个 segment。

```
FlatLayout {
    row_count:  u64,
    dtype:      DType,
    segment_id: SegmentId,     // 指向 segment 映射表
    array_tree: Option<ByteBuffer>,  // 可选：编码树（避免读 segment 才能知道编码）
}
```

- 不可再分，没有 children
- 写入时：`chunk.serialize() → segment_sink.write() → segment_id`
- 读取时：`segment_source.request(segment_id) → deserialize → array`

### 4.2 ChunkedLayout（分块）

将数据按行分成多个 chunk，每个 chunk 是一个子 layout。

```
ChunkedLayout {
    row_count:     u64,
    dtype:         DType,
    chunk_offsets: [u64],          // 每个 chunk 的起始行偏移
    children:      [LayoutRef],    // 子 layout（通常是 FlatLayout）
}
```

- children 按行范围排列，`LayoutChildType::Chunk((idx, row_offset))`
- 读取时根据请求的行范围定位到对应的 children，只读需要的 chunk

### 4.3 StructLayout（列拆分）

将 struct 类型拆成多个 field，每个 field 是一个子 layout。

```
StructLayout {
    row_count: u64,
    dtype:     DType,
    children:  [LayoutRef],    // [validity (optional), field_0, field_1, ...]
}
```

- 如果 struct 是 nullable，第一个 child 是 validity（null bitmap），类型为 `Auxiliary("validity")`
- 后续 children 按 field name，类型为 `Field(name)`
- 自身不持有 segment，数据全在 children 里

### 4.4 ZonedLayout（Zone Map 统计）

在数据之上附加统计信息（min/max 等），支持基于 filter 跳过整个 zone。

```
ZonedLayout {
    row_count:     u64,
    dtype:         DType,
    zone_len:      usize,          // 每个 zone 的行数
    present_stats: Set<Stat>,      // 存储了哪些统计类型
    children:      [LayoutRef; 2], // [data_layout, zone_map_layout]
}
```

- `data_layout`：实际数据
- `zone_map_layout`：统计信息（每个 zone 一行 min/max/null_count 等）
- 读取时先读 zone map → 根据 filter 判断哪些 zone 可以跳过 → 只读命中的 zone

### 4.5 DictLayout（字典编码）

将数据拆为字典表 + 索引数组。

```
DictLayout {
    row_count:   u64,
    dtype:       DType,
    codes_ptype: PType,            // 索引类型（通常 U16）
    children:    [LayoutRef; 2],   // [values_layout, codes_layout]
}
```

- `values_layout`：字典表（唯一值），通常较小
- `codes_layout`：索引数组（引用字典中的位置），与原始数据等行
- 字典跨多个 chunk 共享，直到超限（max_bytes: 1MB 或 max_len: 65535）才 reset

## 5. 写入策略管线（LayoutStrategy）

### 5.1 Strategy 接口

```rust
pub trait LayoutStrategy: 'static + Send + Sync {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,    // 输入：chunk 流
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef>;            // 输出：layout 树节点

    fn buffered_bytes(&self) -> u64;
}
```

每个 strategy 是一个**异步流转换器**：接收 chunk 流 → 转换/重组 → 传给子 strategy → 最终产出 layout 树节点。

### 5.2 内置 Strategy 说明

| Strategy | 作用 | 输入 → 输出 |
|----------|------|-------------|
| **StructStrategy** | 拆列 | struct chunk 流 → 多个列流（并行写入） |
| **RepartitionStrategy** | 按行数/字节数重新分块 | chunk 流 → 固定大小的 chunk 流 |
| **ZonedStrategy** | 计算 zone map 统计 | chunk 流 → 数据 + zone map 统计 |
| **DictStrategy** | 字典编码（条件分支） | chunk 流 → codes 流 + values / 或 fallback |
| **CompressingStrategy** | BtrBlocks 压缩 | chunk → 压缩后的 chunk |
| **BufferedStrategy** | 字节级缓冲，避免小 chunk | chunk 流 → 缓冲合并后的 chunk 流 |
| **ChunkedLayoutStrategy** | 收集多 chunk 为 ChunkedLayout | chunk 流 → ChunkedLayout(children: [FlatLayout...]) |
| **CollectStrategy** | 全量收集为单 chunk | 多 chunk 流 → 单 chunk |
| **FlatLayoutStrategy** | 叶节点，序列化写入 segment | 单 chunk → FlatLayout(segment_id) |

### 5.3 默认策略管线（v1 columnar-first）

来源：`vortex-file/src/strategy.rs` 的 `WriteStrategyBuilder::build()`

```
StructStrategy                                              ← 0. 拆列
│
├─ [data columns] → RepartitionStrategy                     ← 1. 按 8192 行分块
│   │                (min_bytes=0, ×8192行, canonicalize=false)
│   │
│   └─ ZonedStrategy (block_size=8192)                      ← 2. Zone Map
│        │
│        ├─ [data] → DictStrategy                           ← 3. 字典编码（条件分支）
│        │    │
│        │    ├─ [codes] ──→ RepartitionStrategy             ← 4. 合并 (≥1MB, ×8192行)
│        │    │               └─ CompressingStrategy(BtrBlocks, dict=true)
│        │    │                    └─ BufferedStrategy(2MB)
│        │    │                         └─ ChunkedLayoutStrategy
│        │    │                              └─ FlatLayoutStrategy
│        │    │
│        │    ├─ [values] → CompressingStrategy(BtrBlocks, dict=false) → FlatLayoutStrategy
│        │    │
│        │    └─ [fallback] → (同 codes 管线)
│        │
│        └─ [zone map] → CompressingStrategy(BtrBlocks, dict=false) → FlatLayoutStrategy
│
└─ [validity] → CollectStrategy
     └─ CompressingStrategy(BtrBlocks, dict=false) → FlatLayoutStrategy
```

产出的 Layout 树（以 3 列 non-nullable 为例）：

```
StructLayout
  ├─ ZonedLayout (col_0)
  │    ├─ ChunkedLayout [FlatLayout, FlatLayout, ...]   ← 数据
  │    └─ FlatLayout                                     ← zone map
  ├─ ZonedLayout (col_1)
  │    ├─ ChunkedLayout [FlatLayout, FlatLayout, ...]
  │    └─ FlatLayout
  └─ ZonedLayout (col_2)
       ├─ ChunkedLayout [FlatLayout, FlatLayout, ...]
       └─ FlatLayout
```

## 6. 读取路径

### 6.1 文件打开

```
1. 读文件尾部（默认 1MB）
2. 解析 EOF marker → postscript 位置
3. 解析 postscript → 各区域 offset
4. 读 DType、Layout、Statistics、Footer
5. 还原 Layout 树 + Segment 映射表
6. 构造 VortexFile 对象
```

### 6.2 ScanBuilder 读取

```rust
let file = open_options.open(read_source).await?;
let scan = file.scan()?
    .with_projection(select(["col_a", "col_b"], root()))
    .with_filter(gt(column("col_a"), lit(100)))
    .with_row_range(0..10000);
let stream = scan.into_array_stream();
```

### 6.3 读取分发

```
ScanBuilder 配置 projection/filter/row_range
  → LayoutReader 从 layout 树根节点开始
    → StructReader: 只读需要的列
      → ZonedReader: 先读 zone map，跳过不匹配的 zone
        → ChunkedReader: 定位到目标行范围的 chunk
          → FlatReader: segment_source.request(segment_id)
            → 反序列化 → 返回 array
```

### 6.4 LayoutReader 接口

```rust
pub trait LayoutReader {
    // 注册数据分割点（行范围边界）
    fn register_splits(&self, ...);

    // 剪枝评估：filter 是否一定为 false（可以跳过整个节点）
    fn pruning_evaluation(&self, row_range, filter, mask) -> MaskFuture;

    // 过滤评估：应用 filter 产出行掩码
    fn filter_evaluation(&self, row_range, filter, mask) -> MaskFuture;

    // 投影评估：加载选中的列数据
    fn projection_evaluation(&self, row_range, projection, mask) -> ArrayFuture;
}
```

## 7. 最小 IO 分析

### 7.1 读取粒度

| 操作 | 最小 IO | 说明 |
|------|---------|------|
| 打开文件 | 1 次（尾部 1MB） | 读 footer + postscript |
| 读 zone map | 1 次 / zone map segment | 每列一个 zone map segment |
| 读数据 | 1 次 / segment | 一个 segment = 一个 FlatLayout = 一列在一个分块中的数据 |
| 点查单行 | 至少 1 次 / 列 | 必须读包含该行的整个 segment |

### 7.2 点查 IO 放大

假设 row_group_size = 8192，查询 1 行 1 列：

```
实际需要: ~几十字节
实际读取: 1 个 segment ≈ 8192 行该列的压缩数据
IO 放大: ~数千倍
```

### 7.3 根本约束

```rust
// SegmentSource 只支持整 segment 读取
pub trait SegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture;
    // 没有 request_range(id, offset, len) 方法
}
```

**无法在 segment 内部做 range read。** 减少 IO 放大的手段：
- 调小 row_group_size → 更小的 segment → 更少的读放大（但压缩率降低）
- 上游改造：给 SegmentSource 增加 range read 支持（需配合编码层支持偏移计算）

## 8. Sequencing 机制

写入时多个 strategy 并行处理不同列/chunk，需要保证 segment 写入顺序确定性。

**SequenceId**：层级化 ID，支持字典序比较。

```
[0] < [0,0] < [0,1] < [1] < [1,0]
```

**SequencePointer**：生成递增的兄弟 ID。

```rust
let mut ptr = sequence_id.descend();  // 进入子级
let id_0 = ptr.advance();   // [parent, 0]
let id_1 = ptr.advance();   // [parent, 1]
```

**BufferedSegmentSink** 在写入前调用 `collapse()` 等待所有更小 ID 的 segment 完成，保证物理写入顺序与逻辑顺序一致。

## 9. 关键源码文件

| 模块 | 文件 | 说明 |
|------|------|------|
| 文件格式 | `vortex-file/src/lib.rs` | 格式规范文档 |
| Footer | `vortex-file/src/footer/serializer.rs` | Footer 写入 |
| Footer | `vortex-file/src/footer/deserializer.rs` | Footer 读取 |
| Postscript | `vortex-file/src/footer/postscript.rs` | EOF 结构 |
| 写入策略 | `vortex-file/src/strategy.rs` | 默认策略管线 |
| Layout trait | `vortex-layout/src/layout.rs` | Layout 核心抽象 |
| FlatLayout | `vortex-layout/src/layouts/flat/` | 叶节点 layout |
| StructLayout | `vortex-layout/src/layouts/struct_/` | 列拆分 layout |
| ChunkedLayout | `vortex-layout/src/layouts/chunked/` | 分块 layout |
| ZonedLayout | `vortex-layout/src/layouts/zoned/` | Zone Map layout |
| DictLayout | `vortex-layout/src/layouts/dict/` | 字典编码 layout |
| LayoutStrategy | `vortex-layout/src/strategy.rs` | Strategy trait |
| SegmentSource | `vortex-layout/src/segments/source.rs` | 读取接口 |
| SegmentSink | `vortex-layout/src/segments/sink.rs` | 写入接口 |
| Layout 序列化 | `vortex-layout/src/flatbuffers.rs` | Layout 树 FlatBuffer |
| Array 序列化 | `vortex-array/src/serde.rs` | Segment 内部格式 |
| Sequence | `vortex-layout/src/sequence.rs` | 写入顺序控制 |
| VortexFile | `vortex-file/src/file.rs` | 文件 API |
| ScanBuilder | `vortex-scan/src/scan_builder.rs` | 读取 API |
