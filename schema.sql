-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Projects table
create table projects (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Files table
create table files (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references projects(id) on delete cascade not null,
  name text not null,
  path text not null,
  summary text,
  metadata jsonb,
  processed_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- File Chunks table (stores text segments and embeddings)
create table file_chunks (
  id uuid primary key default gen_random_uuid(),
  file_id uuid references files(id) on delete cascade not null,
  content text not null,
  chunk_index integer not null,
  embedding vector(1024) -- Adjust dimension based on model (e.g., qwen3-embedding:0.6b is 1024)
);

-- Keywords table
create table keywords (
  id uuid primary key default gen_random_uuid(),
  keyword text not null unique
);

-- File Keywords (Many-to-Many)
create table file_keywords (
  file_id uuid references files(id) on delete cascade not null,
  keyword_id uuid references keywords(id) on delete cascade not null,
  primary key (file_id, keyword_id)
);

-- Function to match documents
create or replace function match_file_chunks (
  query_embedding vector(1024),
  match_threshold float,
  match_count int,
  filter_project_id uuid
)
returns table (
  id uuid,
  file_id uuid,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    file_chunks.id,
    file_chunks.file_id,
    file_chunks.content,
    1 - (file_chunks.embedding <=> query_embedding) as similarity
  from file_chunks
  join files on files.id = file_chunks.file_id
  where 1 - (file_chunks.embedding <=> query_embedding) > match_threshold
  and files.project_id = filter_project_id
  order by file_chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
