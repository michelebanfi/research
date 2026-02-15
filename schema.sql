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
  embedding vector(1024), -- Your Ollama nomic-embed-text outputs 1024 dimensions
  metadata jsonb default '{}'::jsonb, -- REQ-02: Granular metadata (page, section, is_table, is_reference, etc.)
  is_reference boolean default false, -- REQ-04: Flag for bibliography/reference sections
  parent_chunk_id uuid references file_chunks(id), -- Hierarchical: Parent chunk
  chunk_level integer default 0 -- Hierarchical: 0=root/document, 1=section, 2=paragraph, etc.
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

-- Function to match documents (excludes reference sections by default)
drop function if exists match_file_chunks(vector, float, int, uuid, boolean);

create or replace function match_file_chunks (
  query_embedding vector(1024),
  match_threshold float,
  match_count int,
  filter_project_id uuid,
  include_references boolean default false
)
returns table (
  id uuid,
  file_id uuid,
  content text,
  similarity float,
  parent_chunk_id uuid,
  chunk_level integer
)
language plpgsql
as $$
begin
  return query
  select
    file_chunks.id,
    file_chunks.file_id,
    file_chunks.content,
    1 - (file_chunks.embedding <=> query_embedding) as similarity,
    file_chunks.parent_chunk_id,
    file_chunks.chunk_level
  from file_chunks
  join files on files.id = file_chunks.file_id
  where 1 - (file_chunks.embedding <=> query_embedding) > match_threshold
  and files.project_id = filter_project_id
  and (include_references = true or file_chunks.is_reference = false) -- REQ-04: Filter references
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
-- Function to get related files based on shared nodes (was keywords)
create or replace function get_related_files(p_file_id uuid)
returns table (id uuid, name text, summary text, shared_count bigint)
language plpgsql as $$
begin
  return query
  with my_nodes as (
    select node_id from files_nodes where files_nodes.file_id = p_file_id
  ),
  related_counts as (
    select fn.file_id, count(fn.node_id) as cnt
    from files_nodes fn
    join my_nodes mn on fn.node_id = mn.node_id
    where fn.file_id != p_file_id
    group by fn.file_id
  )
  select f.id, f.name, f.summary, rc.cnt
  from related_counts rc
  join files f on f.id = rc.file_id
  order by rc.cnt desc
  limit 5;
end;
$$;

-- Nodes Table (Entities like 'Process', 'Tool', 'Metric', 'Person')
create table nodes (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  type text, -- 'concept', 'person', 'location', etc.
  properties jsonb DEFAULT '{}'::jsonb,
  unique (name, type)
);

-- Edges Table (Relationships)
create table edges (
  id uuid primary key default gen_random_uuid(),
  source_node_id uuid references nodes(id) on delete cascade not null,
  target_node_id uuid references nodes(id) on delete cascade not null,
  type text not null, -- 'cites', 'uses', 'is_part_of'
  properties jsonb DEFAULT '{}'::jsonb,
  unique (source_node_id, target_node_id, type)
);

-- Junction table to link Files to Nodes (Concepts mentioned in file)
create table files_nodes (
  file_id uuid references files(id) on delete cascade not null,
  node_id uuid references nodes(id) on delete cascade not null,
  primary key (file_id, node_id)
);

-- Recursive CTE to traverse the graph
-- Finds nodes connected to a start_node up to max_depth
create or replace function get_graph_traversal(
  start_node_id uuid,
  max_depth int default 2
)
returns table (
  source_id uuid,
  target_id uuid,
  edge_type text,
  depth int,
  source_name text,
  target_name text
)
language plpgsql
as $$
begin
  return query
  with recursive traversal as (
    -- Base case: Depth 1
    select
      e.source_node_id,
      e.target_node_id,
      e.type as edge_type,
      1 as depth
    from edges e
    where e.source_node_id = start_node_id
    
    union all
    
    -- Recursive step
    select
      e.source_node_id,
      e.target_node_id,
      e.type,
      t.depth + 1
    from edges e
    join traversal t on e.source_node_id = t.target_node_id
    where t.depth < max_depth
  )
  select distinct
    t.source_node_id,
    t.target_node_id,
    t.edge_type,
    t.depth,
    n1.name as source_name,
    n2.name as target_name
  from traversal t
  join nodes n1 on n1.id = t.source_node_id
  join nodes n2 on n2.id = t.target_node_id;
end;
$$;

-- Function to get simple 1-hop neighborhood for a file's linked nodes
-- Used for the generic "Explore Graph" view
create or replace function get_file_graph(p_file_id uuid)
returns table (
  source_name text,
  target_name text,
  edge_type text,
  source_type text,
  target_type text
)
language plpgsql
as $$
begin
  return query
  with file_concepts as (
    select node_id from files_nodes where file_id = p_file_id
  )
  select
    n1.name as source_name,
    n2.name as target_name,
    e.type as edge_type,
    n1.type as source_type,
    n2.type as target_type
  from edges e
  join nodes n1 on e.source_node_id = n1.id
  join nodes n2 on e.target_node_id = n2.id
  where e.source_node_id in (select node_id from file_concepts)
  or e.target_node_id in (select node_id from file_concepts);
end;
$$;

-- Function to get the global graph for a project
-- Returns all edges where at least one node is connected to a file in the project
create or replace function get_project_graph(p_project_id uuid, p_limit int default 500)
returns table (
  source_name text,
  target_name text,
  edge_type text,
  source_type text,
  target_type text
)
language plpgsql
as $$
begin
  return query
  with project_files as (
      select id from files where project_id = p_project_id
  ),
  project_nodes as (
      select distinct node_id from files_nodes where file_id in (select id from project_files)
  )
  select
    n1.name as source_name,
    n2.name as target_name,
    e.type as edge_type,
    n1.type as source_type,
    n2.type as target_type
  from edges e
  join nodes n1 on e.source_node_id = n1.id
  join nodes n2 on e.target_node_id = n2.id
  where e.source_node_id in (select node_id from project_nodes)
     or e.target_node_id in (select node_id from project_nodes)
  limit p_limit;
end;
$$;

-- REQ-DATA-01 + REQ-PERF-01: Bulk graph storage with type-aware entity resolution
-- Uses set-based operations instead of iterative loops
create or replace function store_graph_data(
  p_file_id uuid,
  p_nodes jsonb, -- Array of {name, type}
  p_edges jsonb  -- Array of {source, target, relation, source_type, target_type}
)
returns void language plpgsql as $$
begin
  -- 1. Bulk upsert all nodes using set-based operations
  with node_data as (
    select 
      n->>'name' as name,
      coalesce(n->>'type', 'Concept') as type
    from jsonb_array_elements(p_nodes) as n
    where n->>'name' is not null and trim(n->>'name') != ''
  ),
  inserted_nodes as (
    insert into nodes (name, type)
    select distinct name, type from node_data
    on conflict (name, type) do nothing
    returning id, name, type
  ),
  all_nodes as (
    -- Combine newly inserted with existing nodes
    select id, name, type from inserted_nodes
    union
    select n.id, n.name, n.type from nodes n
    inner join node_data nd on n.name = nd.name and n.type = nd.type
  )
  -- 2. Bulk link nodes to file
  insert into files_nodes (file_id, node_id)
  select distinct p_file_id, an.id from all_nodes an
  on conflict do nothing;

  -- 3. Bulk insert edges using type-aware resolution (REQ-DATA-01)
  with edge_data as (
    select
      e->>'source' as source_name,
      e->>'target' as target_name,
      coalesce(e->>'relation', 'related_to') as relation,
      coalesce(e->>'source_type', 'Concept') as source_type,
      coalesce(e->>'target_type', 'Concept') as target_type
    from jsonb_array_elements(p_edges) as e
    where e->>'source' is not null and e->>'target' is not null
  ),
  resolved_edges as (
    select
      src.id as source_id,
      tgt.id as target_id,
      ed.relation
    from edge_data ed
    -- REQ-DATA-01: Resolve by BOTH name AND type
    left join nodes src on src.name = ed.source_name and src.type = ed.source_type
    left join nodes tgt on tgt.name = ed.target_name and tgt.type = ed.target_type
    where src.id is not null and tgt.id is not null
  )
  insert into edges (source_node_id, target_node_id, type)
  select distinct source_id, target_id, relation from resolved_edges
  on conflict (source_node_id, target_node_id, type) do nothing;
end;
$$;


-- REQ-SEARCH-01: Full-Text Search (FTS) support
-- Add tsvector column to file_chunks if it doesn't exist
alter table file_chunks add column if not exists fts tsvector;

-- Create index for FTS
create index if not exists file_chunks_fts_idx on file_chunks using gin (fts);

-- Trigger to automatically update fts column on content change
create or replace function file_chunks_tsvector_trigger() returns trigger as $$
begin
  new.fts := to_tsvector('english', new.content);
  return new;
end
$$ language plpgsql;

drop trigger if exists tsvectorupdate on file_chunks;
create trigger tsvectorupdate before insert or update
on file_chunks for each row execute function file_chunks_tsvector_trigger();

-- Backfill existing chunks (optional, but good practice)
update file_chunks set fts = to_tsvector('english', content) where fts is null;


-- REQ-PERF-02: Optimized Composite Search RPC (Fixes N+1 problem)
-- Returns chunk + file metadata + similarity + keyword rank in ONE call
drop function if exists search_file_chunks_rpc(vector, float, int, uuid, boolean, text);

create or replace function search_file_chunks_rpc (
  query_embedding vector(1024),
  match_threshold float,
  match_count int,
  filter_project_id uuid,
  include_references boolean default false,
  keyword_query text default null
)
returns table (
  id uuid,
  file_id uuid,
  content text,
  similarity float,
  metadata jsonb,
  file_path text,
  file_name text,
  chunk_level integer,
  rank float -- Combined rank
)
language plpgsql
as $$
begin
  return query
  select
    fc.id,
    fc.file_id,
    fc.content,
    1 - (fc.embedding <=> query_embedding) as similarity,
    fc.metadata,
    f.path as file_path,
    f.name as file_name,
    fc.chunk_level,
    (
      (1 - (fc.embedding <=> query_embedding)) * 0.7 + -- Semantic weight
      (case when keyword_query is not null then ts_rank(fc.fts, plainto_tsquery('english', keyword_query)) else 0 end) * 0.3 -- Keyword weight
    ) as rank
  from file_chunks fc
  join files f on f.id = fc.file_id
  where 1 - (fc.embedding <=> query_embedding) > match_threshold
  and f.project_id = filter_project_id
  and (include_references = true or fc.is_reference = false)
  and (keyword_query is null or fc.fts @@ plainto_tsquery('english', keyword_query))
  order by rank desc
  limit match_count;
end;
$$;

