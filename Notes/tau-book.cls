%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LaTeX class ~ 'tau-book.cls'
% Version 2.2 (15/03/2024)
%
% Description: 
% The LaTeX class "tau-book.cls" contains all the settings, 
% packages and commands for producing lab reports or academic 
% papers. Due to its clean and structured code, users can 
% easily customize this class to their needs and preferences. 
% The class integrates BibLaTeX to manage the bibliography.
% 
% Author: 
% Guillermo Jimenez (memo.notess1@gmail.com)
%
% Note:
% You may modify 'tau-book.cls' file according to your
% needs and preferences. This LaTeX class file defines 
% the document layout, formatting, and behavior.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        WARNING:
% It is important to proceed with caution and ensure that 
% any modifications made are compatible with LaTeX 
% syntax and conventions to avoid errors or unexpected 
% behavior in the document compilation process.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------------------
% Article class requirements
%----------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tau-book}[2024/03/15 Tau-book LaTeX class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}
\ProcessOptions\relax
\LoadClass[twocolumn]{extarticle}

%----------------------------------------------------------
% Required packages
%----------------------------------------------------------

\RequirePackage[utf8]{inputenc}
\RequirePackage{etoolbox}
\RequirePackage[framemethod=tikz]{mdframed}
\RequirePackage{titling}
\RequirePackage{lettrine}
\RequirePackage{microtype}
\RequirePackage[bottom,hang,flushmargin,ragged,symbol*]{footmisc}
\RequirePackage{abstract}
\RequirePackage{fancyhdr}
\RequirePackage{xifthen}
\usepackage{adjustbox}
\usepackage{adforn}
\usepackage{lastpage}
\RequirePackage[explicit]{titlesec}
\RequirePackage{caption}
\RequirePackage{setspace}
\RequirePackage{iflang}
\RequirePackage{listings}

%----------------------------------------------------------
% Font used
%----------------------------------------------------------

\usepackage{stix2}

%----------------------------------------------------------
% Graphics path and lengths
%----------------------------------------------------------

\setlength{\columnsep}{15pt}
\graphicspath{{Figures/}{./}} 
% \renewcommand{\baselinestretch}{1}    % To change the line spacing

%----------------------------------------------------------
% New commands
%----------------------------------------------------------

\newcommand{\professor}[1]{\def\@professor{#1}}
\newcommand{\keywords}[1]{\def\@keywords{#1}}

%----------------------------------------------------------
% Geometry package setup
%----------------------------------------------------------

\RequirePackage[
	left=1.25cm,
	right=1.25cm,
	top=2cm,
	bottom=2cm,
    headsep=0.75cm,
	% showframe
]{geometry}

%----------------------------------------------------------
% Header and footer 
%----------------------------------------------------------

\newcommand{\footerfont}{\normalfont\fontsize{7}{9}\selectfont}
\newcommand{\theday}[1]{\renewcommand{\theday}{#1}}
\newcommand{\institution}[1]{\def\@institution{#1}}
\newcommand{\ftitle}[1]{\def\@ftitle{#1}}
\newcommand{\etal}[1]{\def\@etal{#1}}
\newcommand{\course}[1]{\def\@course{#1}}
\pagestyle{fancy}

\pagenumbering{Roman}

\fancypagestyle{firststyle}{
   \fancyfoot[R]{\footerfont\@institution\hspace{7pt}|\hspace{7pt}\textbf{\theday}\hspace{7pt}|\hspace{7pt}\@ftitle\hspace{7pt}|\hspace{7pt}\textbf{\thepage\textendash\pageref{LastPage}}}
   \fancyfoot[L]{\footerfont\@ifundefined{@course}{}{\@course}}
   \fancyhead[C]{}
}

% Header
\fancyhead[C]{\footerfont\it\@title}
\fancyhead[LE,RO]{}
\fancyhead[LO,RE]{}

% Footer
\lfoot{}%
\cfoot{}%
\rfoot{}%

\fancyfoot[LE]{\footerfont\textbf{\thepage}\hspace{7pt}|\hspace{7pt}\@ifundefined{@course}{}{\@course}}
\fancyfoot[RO]{\footerfont\@institution\hspace{7pt}|\hspace{7pt}\textbf{\theday}\hspace{7pt}|\hspace{7pt}\@ftitle\hspace{7pt} |\hspace{7pt}\textbf{\thepage}}
\fancyfoot[RE,LO]{\footerfont\@ifundefined{@etal}{}{\@etal}\ifnum \value{authors} > 1\hspace{5pt}\textit{}\fi}

\renewcommand{\headrulewidth}{0pt}% % No header rule
\renewcommand{\footrulewidth}{0pt}% % No footer rule

%----------------------------------------------------------
% Tau start (lettrine)
%----------------------------------------------------------

\newcommand{\dropcapfont}{\color{taublue}\bfseries\fontsize{26pt}{28pt}\selectfont}
\newcommand{\taustart}[1]{\lettrine[lines=2,lraise=0,findent=2pt, nindent=0em]{{\dropcapfont{#1}}}{}}

%----------------------------------------------------------
% Abstract preferences
%----------------------------------------------------------

% Abstract font
\renewcommand{\abstractnamefont}{\selectfont\small\itshape}

% Abstract text font
\renewcommand{\abstracttextfont}{\selectfont\small\itshape}

% Custom abstract
\renewenvironment{abstract}         
    {\selectfont\abstractnamefont\abstracttextfont{\color{taublue}\bfseries\itshape\abstractname---}\ignorespaces
}{
    \@ifundefined{@keywords}{}
        {\selectfont\newline\par{\color{taublue}\itshape\bfseries\small keywords---}\ignorespaces\@keywords}
}

%----------------------------------------------------------
% Title preferences
%----------------------------------------------------------
% Note ~ If you like to uncomment the lines (rule) to 
% customize the title, set to 15pt the last \vskip.
%----------------------------------------------------------

\setlength{\droptitle}{-25\baselineskip} % Move the title up

\renewcommand{\@maketitle}{%
        \vskip-24pt
    % {\color{taublue}\rule{\linewidth}{0.5pt}}
    %	\vskip3pt
    {\centering\bfseries\LARGE\color{taublue}\@title\par}
		\vskip8pt
    {\centering\normalsize\@author\par}
        \vskip8pt
    {\centering\fontsize{7pt}{8pt}\selectfont\@professor\par}
    %	\vskip3pt
    % {\color{taublue}\rule{\linewidth}{0.5pt}}
        \vskip24pt						% 15pt
}%

%----------------------------------------------------------
% Affil setup
%----------------------------------------------------------

\usepackage{authblk} % For custom environment affiliation

\newcommand{\andlanguage}{
    \iflanguage{spanish}{
        { y }%
    }{%
        { and }%
    }%
}

\setlength{\affilsep}{8pt}
\renewcommand\Authfont{\normalfont\bfseries\fontsize{9}{11}}
\renewcommand\Affilfont{\normalfont\fontsize{7}{8}}
\renewcommand\AB@affilsepx{; \protect\Affilfont}
\makeatletter
\renewcommand\Authands{, }
\renewcommand\Authand{\ignorespaces\andlanguage}

%----------------------------------------------------------
% Section formating
%----------------------------------------------------------

\setcounter{secnumdepth}{5}

\titleformat{\section}
  	{\color{taublue}\large\bfseries}
  	{\thesection.}
  	{0.5em}
  	{#1}
  	[]
  
\titleformat{name=\section,numberless}
  	{\color{taublue}\large\bfseries}
  	{}
  	{0em}
  	{#1}
  	[]  
  
\titleformat{\subsection}[block] % You may change it to "runin"
  	{\color{taublue}\bfseries}
  	{\thesubsection.}
  	{0.5em}
  	{#1}      % If using runin, change #1 to '#1. ' (space at the end)
  	[]
  
\titleformat{\subsubsection}[block] % You may change it to "runin"
  	{\color{taublue}\small\bfseries\itshape}
  	{\thesubsubsection.}
  	{0.5em}
  	{#1}      % If using runin, change #1 to '#1. ' (space at the end)
  	[]    
  
\titleformat{\paragraph}[runin]
	{\small\bfseries}
  	{}
  	{0em}
  	{#1} 
  
\titlespacing*{\section}{0pc}{3ex \@plus4pt \@minus3pt}{5pt}
\titlespacing*{\subsection}{0pc}{2.5ex \@plus3pt \@minus2pt}{2pt}
\titlespacing*{\subsubsection}{0pc}{2ex \@plus2.5pt \@minus1.5pt}{2pt}
\titlespacing*{\paragraph}{0pc}{1.5ex \@plus2pt \@minus1pt}{12pt}

%----------------------------------------------------------
% Table of contents
%----------------------------------------------------------

\newlength{\tocsep} 
\setlength\tocsep{1.5pc} % Sets the indentation of the sections in the table of contents
\setcounter{tocdepth}{5} % Three levels in the table of contents section: sections, subsections and subsubsections

\usepackage{titletoc}
\contentsmargin{0cm}

\titlecontents{section}[\tocsep]
    {\addvspace{4pt}\selectfont\bfseries}
    {\contentslabel[\thecontentslabel]{\tocsep}}
    {}
    {\hfill\thecontentspage}
    []

\titlecontents{subsection}[\tocsep]
    {\addvspace{4pt}\selectfont}
    {\contentslabel[\thecontentslabel]{\tocsep}}
    {}
    {\ \titlerule*[.5pc]{.}\ \thecontentspage}
    []

\titlecontents*{subsubsection}[\tocsep]
    {\footnotesize\selectfont}
    {}
    {}
    {}
    [\ \textbullet\ ]

%----------------------------------------------------------
% Figures caption style
%----------------------------------------------------------

\captionsetup[figure]{format=plain, labelsep=period, textfont={footnotesize, it}, justification=centering, labelfont={footnotesize,bf}}

%----------------------------------------------------------
% Table caption style
%----------------------------------------------------------

\captionsetup*[table]{labelfont+={footnotesize,bf},textfont+={footnotesize,it},skip=10pt,position=below,labelsep=period} % Change it to "above" if you prefer the caption above the table.

%----------------------------------------------------------
% Listings caption style (codes)
%----------------------------------------------------------

\captionsetup[lstlisting]{font=footnotesize, labelfont=bf, textfont=it, belowskip=0pt, position=below, labelsep=period}
\renewcommand\lstlistingname{Code}
\renewcommand\lstlistlistingname{Code}

%----------------------------------------------------------
% Listings defined styles (codes)
%----------------------------------------------------------

% Defined colors for listings environment
\definecolor{taucodeback}{RGB}{248, 248, 248}
\definecolor{taucodecomment}{RGB}{1, 136, 0}
\definecolor{taucodekey}{RGB}{53, 53, 128}
\definecolor{taucodestring}{RGB}{122, 36, 47}
\definecolor{taugray}{RGB}{0.5,0.5,0.5}
\definecolor{tauredmatlab}{RGB}{199, 78, 0}

% Matlab
\lstset{language=Matlab,
    backgroundcolor=\color{taucodeback},   
	commentstyle=\color{taucodecomment},
	keywordstyle=\bf\color{taucodekey},
	numberstyle=\tiny\color{taugray},
	stringstyle=\color{taucodestring},
	basicstyle=\footnotesize\ttfamily,
	breakatwhitespace=false,         
	breaklines=true,                 
	captionpos=b,                    
	keepspaces=true,          		
	numbers=left,                   % if "none" change the values
	numbersep=5pt,                  % 0pt
	showspaces=false,                
	showstringspaces=false,
	showtabs=false,                  
	tabsize=2,
	aboveskip=10pt,
	belowskip=15pt,
	xleftmargin=9pt,				% 0pt
	rulecolor=\color{taugray},
	emph=[1]{for,end,break},emphstyle=[1]\color{tauredmatlab}
}

% General
\lstdefinestyle{taucoding}{
    backgroundcolor=\color{taucodeback},   
    commentstyle=\color{taucodecomment},
    keywordstyle=\bf\color{taucodekey},
    numberstyle=\tiny\color{taugray},
    stringstyle=\color{taucodestring},
    basicstyle=\footnotesize\ttfamily,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,				 	% if "none" change the values  
    numbersep=5pt,    				% 0pt              
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    aboveskip=10pt,
    belowskip=15pt,
    xleftmargin=9pt,				% 0pt
    rulecolor=\color{taugray},
}

\lstset{style=taucoding}

%----------------------------------------------------------
% Footnotes
%----------------------------------------------------------

\definecolor{black50}{gray}{0.5}
\renewcommand*{\footnotelayout}{\normalfont\fontsize{6}{8}}
\renewcommand{\footnoterule}{
 	\kern -3pt
  	{\color{black50} \hrule width 72pt height 0.25pt}
  	\kern 2.5pt
}

%----------------------------------------------------------
% URL settings
%----------------------------------------------------------

\RequirePackage{url}
\RequirePackage{xurl}
\renewcommand\UrlFont{\selectfont}
\RequirePackage[colorlinks=true, allcolors=taublue]{hyperref}

%----------------------------------------------------------
% BiblaTeX
%----------------------------------------------------------

\usepackage[
    backend=biber,
    style=ieee,
    sorting=ynt
]{biblatex}

\addbibresource{tau.bib}

%----------------------------------------------------------

\endinput