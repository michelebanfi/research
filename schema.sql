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

-- Function to link keywords to a file (Batch Upsert)
create or replace function link_file_keywords(p_file_id uuid, p_keywords text[])
returns void language plpgsql as $$
begin
  with inserted_keywords as (
    insert into keywords (keyword)
    select distinct unnest(p_keywords)
    on conflict (keyword) do nothing
    returning id
  ),
  existing_keywords as (
    select id from keywords where keyword = any(p_keywords)
  ),
  all_keyword_ids as (
    select id from inserted_keywords
    union
    select id from existing_keywords
  )
  insert into file_keywords (file_id, keyword_id)
  select p_file_id, id from all_keyword_ids
  on conflict do nothing;
end;
$$;

-- Function to get related files based on shared keywords
create or replace function get_related_files(p_file_id uuid)
returns table (id uuid, name text, summary text, shared_count bigint)
language plpgsql as $$
begin
  return query
  with my_keywords as (
    select keyword_id from file_keywords where file_keywords.file_id = p_file_id
  ),
  related_counts as (
    select fk.file_id, count(fk.keyword_id) as cnt
    from file_keywords fk
    join my_keywords mk on fk.keyword_id = mk.keyword_id
    where fk.file_id != p_file_id
    group by fk.file_id
  )
  select f.id, f.name, f.summary, rc.cnt
  from related_counts rc
  join files f on f.id = rc.file_id
  order by rc.cnt desc
  limit 5;
end;
$$;
