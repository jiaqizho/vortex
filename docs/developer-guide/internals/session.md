# Session and Registries

A `VortexSession` is a type-indexed map that holds all the runtime state for the Vortex
ecosystem. Each major component of Vortex defines its own session variable type containing a
registry of plugins. There is typically one session per process, and it is passed explicitly
through the API rather than relying on global or thread-local state.

Plugins can add state while constructing the session, for example to hold caches or other
shared resources.

## Design

The session is built on two primitives from the `vortex-session` crate:

- **`VortexSession`** -- a cloneable, thread-safe map from Rust `TypeId` to a boxed value. Any
  type implementing `SessionVar` and `Clone` can be stored as a session variable.
- **`Registry<T>`** -- a concurrent map from string IDs to values of type `T`, used by each
  component to look up registered plugins at runtime.

`VortexSession` is backed by an immutable `Arc<HashMap>`, so cloning is cheap and reading
components takes no session-map lock. Its set of components is fixed after construction.
To change configuration, use `to_builder()` and build a new session. Configuration fields
belong to the new session, while registries and caches can retain shared backing storage
according to their `Clone` implementations.

## Component Registries

Each Vortex crate defines a session variable that holds a registry for its extension points:

| Session Variable  | Crate            | Registry Contents                            |
|-------------------|------------------|----------------------------------------------|
| `DTypeSession`    | `vortex-array`   | Extension dtype vtables (Date, Time, ...)    |
| `ArraySession`    | `vortex-array`   | Array encoding vtables (ALP, FSST, ...)      |
| `ScalarFnSession` | `vortex-array`   | Scalar function vtables                      |
| `LayoutSession`   | `vortex-layout`  | Layout encoding vtables (Flat, Chunked, ...) |
| `RuntimeSession`  | `vortex-io`      | Async runtime handle                         |
| `CudaSession`     | `vortex-cuda`    | CUDA context, kernels, and stream pool       |

Register required session variables before building a session. Their `Default` implementations
register the built-in plugins for each component. `get()` panics if a component is missing;
`get_opt()` returns `None` and does not initialize it. Mutable plugin registries inside existing
components still support registration after the session has been built.

## Registering Plugins

Plugins register with the session by accessing the relevant component and calling `register`:

```rust
// Register a custom array encoding
session.arrays().register(MyEncoding);

// Register a custom layout
session.layouts().register(MyLayout::encoding());

// Register a custom scalar function
session.scalar_fns().register(MyScalarFnVTable);
```

Crates that bundle multiple plugins typically expose an `initialize` function that registers
everything at once. The top-level `vortex` crate calls these during `VortexSession::default()`
to register all built-in encodings.

## Explicit Passing

Sessions are passed explicitly through constructors and method arguments. This means every API
that needs access to registries -- file readers, writers, scan builders, layout readers -- receives
the session directly rather than reaching for global state.

```rust
// Opening a file
session.open_options()
    .open(reader)
    .await?;

// Writing a file
session.write_options()
    .write(&mut file, array_stream)
    .await?;

// Scanning a layout
ScanBuilder::new(session.clone(), layout_reader)
    .with_filter(expr)
    .into_array_stream()?;
```

Many APIs use extension traits to provide ergonomic methods directly on the session. For example,
`OpenOptionsSessionExt` adds `.open_options()`. Required components such as `ArraySession`,
`LayoutSession`, and `RuntimeSession` must be registered when constructing the session;
extension traits do not statically enforce their presence.

## Constructing a Session

The `VortexSession::default()` provided by the top-level `vortex` crate constructs a session with
all built-in components and encodings:

```rust
let session = VortexSession::default();
```

For tests or specialized use-cases, start with the standard array components and add layout
and runtime state using the builder:

```rust
let session = vortex_array::array_session()
    .to_builder()
    .with::<LayoutSession>()
    .with::<RuntimeSession>()
    .build();
```

For array execution, `vortex_array::array_session()` supplies the standard array, dtype,
function, optimizer, Arrow conversion, and memory components. Add layout and runtime state
when using it for file operations.
