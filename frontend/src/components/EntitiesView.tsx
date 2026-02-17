import { useMemo } from 'react'
import { 
  Circle, 
  Box, 
  Database, 
  Activity, 
  User,
  Share2
} from 'lucide-react'

interface Entity {
  name: string
  type: string
}

interface EntitiesViewProps {
  entities: Entity[]
}

const TYPE_CONFIG: Record<string, { color: string; bgColor: string; icon: React.ReactNode }> = {
  'Concept': {
    color: 'text-red-500',
    bgColor: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
    icon: <Circle size={14} />
  },
  'Tool': {
    color: 'text-teal-500',
    bgColor: 'bg-teal-50 dark:bg-teal-900/20 border-teal-200 dark:border-teal-800',
    icon: <Box size={14} />
  },
  'System': {
    color: 'text-blue-500',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    icon: <Database size={14} />
  },
  'Metric': {
    color: 'text-green-500',
    bgColor: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
    icon: <Activity size={14} />
  },
  'Person': {
    color: 'text-purple-500',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
    icon: <User size={14} />
  },
  'Class': {
    color: 'text-orange-500',
    bgColor: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800',
    icon: <Box size={14} />
  },
  'Method': {
    color: 'text-cyan-500',
    bgColor: 'bg-cyan-50 dark:bg-cyan-900/20 border-cyan-200 dark:border-cyan-800',
    icon: <Circle size={14} />
  },
  'Function': {
    color: 'text-indigo-500',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800',
    icon: <Circle size={14} />
  }
}

function getTypeConfig(type: string) {
  return TYPE_CONFIG[type] || {
    color: 'text-slate-500',
    bgColor: 'bg-slate-50 dark:bg-slate-900/20 border-slate-200 dark:border-slate-800',
    icon: <Circle size={14} />
  }
}

export default function EntitiesView({ entities }: EntitiesViewProps) {
  const groupedEntities = useMemo(() => {
    const groups: Record<string, Entity[]> = {}
    entities.forEach(entity => {
      if (!groups[entity.type]) {
        groups[entity.type] = []
      }
      groups[entity.type].push(entity)
    })
    return groups
  }, [entities])
  
  if (entities.length === 0) {
    return (
      <div className="text-center py-12 text-muted">
        <Share2 size={48} className="mx-auto mb-4 opacity-50" />
        <p className="text-sm">No entities extracted from this document</p>
      </div>
    )
  }
  
  return (
    <div className="h-full overflow-y-auto p-4">
      {/* Stats summary */}
      <div className="mb-6 flex flex-wrap gap-3">
        {Object.entries(groupedEntities).map(([type, items]) => {
          const config = getTypeConfig(type)
          return (
            <div 
              key={type}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${config.bgColor}`}
            >
              <span className={config.color}>{config.icon}</span>
              <span className="text-sm font-medium">{items.length}</span>
              <span className={`text-xs ${config.color} opacity-80`}>{type}s</span>
            </div>
          )
        })}
      </div>
      
      {/* Entity groups */}
      <div className="space-y-6">
        {Object.entries(groupedEntities).map(([type, items]) => {
          const config = getTypeConfig(type)
          return (
            <div key={type}>
              <div className="flex items-center gap-2 mb-3">
                <span className={config.color}>{config.icon}</span>
                <h4 className="font-medium">{type}s</h4>
                <span className="text-xs text-muted">({items.length})</span>
              </div>
              
              <div className="flex flex-wrap gap-2">
                {items.map((entity, idx) => (
                  <span
                    key={idx}
                    className={`
                      px-3 py-1.5 rounded-lg border text-sm
                      transition-all hover:scale-105 cursor-default
                      ${config.bgColor}
                    `}
                    title={entity.type}
                  >
                    {entity.name}
                  </span>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
