import React from 'react';

interface ResultSectionProps {
  title: string;
  content: string | React.ReactNode;
  variant?: 'default' | 'info';
}

const ResultSection: React.FC<ResultSectionProps> = ({ title, content, variant = 'default' }) => {
  const bgColor = variant === 'info' ? 'bg-blue-50' : 'bg-bgSecondary';
  const borderColor = variant === 'info' ? 'border-blue-500' : 'border-primary';
  const titleColor = variant === 'info' ? 'text-blue-700' : 'text-primary';

  return (
    <div className={`${bgColor} p-5 rounded-lg mb-4 border-l-4 ${borderColor}`}>
      <h3 className={`text-lg font-semibold mb-3 ${titleColor}`}>{title}</h3>
      {typeof content === 'string' ? (
        <p className="leading-relaxed text-textPrimary whitespace-pre-wrap">{content}</p>
      ) : (
        content
      )}
    </div>
  );
};

export default ResultSection;