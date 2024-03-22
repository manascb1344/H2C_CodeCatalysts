import React from 'react';
import { getBezierPath, BaseEdge, useStore } from 'reactflow';

const getSpecialPath = ({ sourceX, sourceY, targetX, targetY }, offset) => {
  const centerX = (sourceX + targetX) / 2;
  const centerY = (sourceY + targetY) / 2;

  return `M ${sourceX},${sourceY} Q ${centerX},${centerY + offset} ${targetX},${targetY}`;
};

const CustomEdge = ({
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
}) => {
  const isBiDirectionEdge = useStore((s) =>
    s.edges.some(
      (e) =>
        (e.source === target && e.target === source) || (e.target === source && e.source === target)
    )
  );

  const edgePaths = [];
  const edgePathParams = {
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  };

  if (isBiDirectionEdge) {
    edgePaths.push(getSpecialPath(edgePathParams, sourceX < targetX ? 25 : -25));
    edgePaths.push(getSpecialPath(edgePathParams, sourceX < targetX ? 50 : -50)); // Add more offsets for multiple edges
  } else {
    edgePaths.push(getBezierPath(edgePathParams));
  }

  return (
    <>
      {edgePaths.map((path, index) => (
        <React.Fragment key={index}>
          <BaseEdge path={path} markerEnd={markerEnd} />
        </React.Fragment>
      ))}
    </>
  );
};

export default CustomEdge;
