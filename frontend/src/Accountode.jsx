import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import 'reactflow/dist/style.css';

const AccountNode = memo(({ data, isConnectable }) => {
  return (
    <div className="border border-gray-500 rounded-lg p-2">
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#555' }}
        isConnectable={isConnectable}
      />
      <div className="account-node">
        <strong>{data.label}</strong>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        id="a"
        style={{ top: '50%', transform: 'translateY(-50%)', background: '#555' }}
        isConnectable={isConnectable}
      />
    </div>
  );
});

export default AccountNode;
