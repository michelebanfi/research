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
  is_reference boolean default false -- REQ-04: Flag for bibliography/reference sections
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

-- Function to store graph data (Batch Upsert)
create or replace function store_graph_data(
  p_file_id uuid,
  p_nodes jsonb, -- Array of {name, type}
  p_edges jsonb  -- Array of {source, target, type}
)
returns void language plpgsql as $$
declare
  node_record jsonb;
  edge_record jsonb;
  src_id uuid;
  tgt_id uuid;
begin
  -- 1. Insert Nodes and link to File
  for node_record in select * from jsonb_array_elements(p_nodes)
  loop
    -- Upsert Node
    with inserted_node as (
      insert into nodes (name, type)
      values (node_record->>'name', node_record->>'type')
      on conflict (name, type) do update set properties = nodes.properties -- dummy update to return id if exists? No, need RETURNING
      returning id
    )
    select id into src_id from inserted_node;
    
    -- If it existed, we need to fetch it
    if src_id is null then
      select id into src_id from nodes where name = node_record->>'name' and type = node_record->>'type';
    end if;

    -- Link to File
    insert into files_nodes (file_id, node_id)
    values (p_file_id, src_id)
    on conflict do nothing;
  end loop;

  -- 2. Insert Edges
  -- Re-loop to find IDs (less efficient but safer given we need IDs)
  for edge_record in select * from jsonb_array_elements(p_edges)
  loop
    select id into src_id from nodes where name = edge_record->>'source' limit 1;
    select id into tgt_id from nodes where name = edge_record->>'target' limit 1;
    
    if src_id is not null and tgt_id is not null then
      insert into edges (source_node_id, target_node_id, type)
      values (src_id, tgt_id, edge_record->>'relation')
      on conflict do nothing;
    end if;
  end loop;
end;
$$;
