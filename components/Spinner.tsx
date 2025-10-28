import React from 'react';

interface SpinnerProps {
  size?: string; // e.g., 'w-10 h-10', 'w-5 h-5'
  color?: string; // e.g., 'border-t-primary', 'border-t-white'
  className?: string; // additional classes
}

const Spinner: React.FC<SpinnerProps> = ({ size = 'w-10 h-10', color = 'border-t-primary', className = '' }) => {
  return (
    <div className={`animate-spin border-4 border-borderColor ${color} rounded-full ${size} ${className}`}></div>
  );
};

export default Spinner;