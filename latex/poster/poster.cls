\ProvidesClass{poster}[2023/06/26 Poster]

\def\posterformat{A0}
\def\posterorientation{portrait}

\DeclareOption{A0}{\def\posterformat{A0}}
\DeclareOption{A1}{\def\posterformat{A1}}
\DeclareOption{A2}{\def\posterformat{A2}}
\DeclareOption{A3}{\def\posterformat{A3}}
\DeclareOption{A4}{\def\posterformat{A4}}
\DeclareOption{portrait}{\def\posterorientation{portrait}}
\DeclareOption{landscape}{\def\posterorientation{landscape}}
\DeclareOption*{%
\def\posterformat{A0}%
\def\posterorientation{portrait}%
}
\ProcessOptions\relax

\RequirePackage{ifthen}

\PassOptionsToClass{25pt}{tikzposter}
\PassOptionsToClass{a0paper}{tikzposter}
\ifthenelse{\equal{\posterorientation}{landscape}}{%
\PassOptionsToClass{landscape}{tikzposter}
}{}
\LoadClass{tikzposter}

\RequirePackage[default]{opensans}
\fontfamily{opensans}\selectfont

\RequirePackage{pgfpages}

\makeatletter
\define@key{pgfpagesuselayoutoption}{A0}[]%
{\def\pgfpageoptionheight{1189mm}\def\pgfpageoptionwidth{841mm}\def\pgfpageoptionmargin{6mm}}
\define@key{pgfpagesuselayoutoption}{A0landscape}[]%
{\def\pgfpageoptionheight{841mm}\def\pgfpageoptionwidth{1189mm}\def\pgfpageoptionmargin{0mm}}
\define@key{pgfpagesuselayoutoption}{A1}[]%
{\def\pgfpageoptionheight{841mm}\def\pgfpageoptionwidth{594mm}\def\pgfpageoptionmargin{5mm}}
\define@key{pgfpagesuselayoutoption}{A1landscape}[]%
{\def\pgfpageoptionheight{594mm}\def\pgfpageoptionwidth{841mm}\def\pgfpageoptionmargin{0mm}}
\define@key{pgfpagesuselayoutoption}{A2}[]%
{\def\pgfpageoptionheight{594mm}\def\pgfpageoptionwidth{420mm}\def\pgfpageoptionmargin{3mm}}
\define@key{pgfpagesuselayoutoption}{A2landscape}[]%
{\def\pgfpageoptionheight{420mm}\def\pgfpageoptionwidth{594mm}\def\pgfpageoptionmargin{0mm}}
\define@key{pgfpagesuselayoutoption}{A3}[]%
{\def\pgfpageoptionheight{420mm}\def\pgfpageoptionwidth{297mm}\def\pgfpageoptionmargin{2.25mm}}
\define@key{pgfpagesuselayoutoption}{A3landscape}[]%
{\def\pgfpageoptionheight{297mm}\def\pgfpageoptionwidth{420mm}\def\pgfpageoptionmargin{0mm}}
\define@key{pgfpagesuselayoutoption}{A4}[]%
{\def\pgfpageoptionheight{297mm}\def\pgfpageoptionwidth{210mm}\def\pgfpageoptionmargin{1.5mm}}
\define@key{pgfpagesuselayoutoption}{A4landscape}[]%
{\def\pgfpageoptionheight{210mm}\def\pgfpageoptionwidth{297mm}\def\pgfpageoptionmargin{0mm}}
\makeatother

\pgfpagesdeclarelayout{resize}
{\def\pgfpageoptionborder{0pt}}
{\pgfpagesphysicalpageoptions{%
logical pages=1,physical height=\pgfpageoptionheight,physical width=\pgfpageoptionwidth}
\pgfpageslogicalpageoptions{1}{%
resized width=\pgfphysicalwidth,resized height=\pgfphysicalheight,border shrink=\pgfpageoptionborder,center=\pgfpoint{\dimexpr0.5\pgfphysicalwidth+\pgfpageoptionmargin\relax}{.5\pgfphysicalheight}}}

\ifthenelse{\equal{\posterformat}{A0}}{%
\ifthenelse{\equal{\posterorientation}{landscape}}{%
\pgfpagesuselayout{resize}[A0landscape]
}{%
\pgfpagesuselayout{resize}[A0]
}
}{}
\ifthenelse{\equal{\posterformat}{A1}}{%
\ifthenelse{\equal{\posterorientation}{landscape}}{%
\pgfpagesuselayout{resize}[A1landscape]
}{%
\pgfpagesuselayout{resize}[A1]
}
}{}
\ifthenelse{\equal{\posterformat}{A2}}{%
\ifthenelse{\equal{\posterorientation}{landscape}}{%
\pgfpagesuselayout{resize}[A2landscape]
}{%
\pgfpagesuselayout{resize}[A2]
}
}{}
\ifthenelse{\equal{\posterformat}{A3}}{%
\ifthenelse{\equal{\posterorientation}{landscape}}{%
\pgfpagesuselayout{resize}[A3landscape]
}{%
\pgfpagesuselayout{resize}[A3]
}
}{}
\ifthenelse{\equal{\posterformat}{A4}}{%
\ifthenelse{\equal{\posterorientation}{landscape}}{%
\pgfpagesuselayout{resize}[A4landscape]
}{%
\pgfpagesuselayout{resize}[A4]
}
}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\makeatletter
\renewcommand\tiny{\@setfontsize\tiny{10}{15}}
\renewcommand\scriptsize{\@setfontsize\scriptsize{15}{20}}
\renewcommand\footnotesize{\@setfontsize\footnotesize{20}{25}}
\renewcommand\small{\@setfontsize\small{25}{30}}
\renewcommand\normalsize{\@setfontsize\normalsize{30}{35}}
\renewcommand\large{\@setfontsize\large{35}{40}}
\renewcommand\Large{\@setfontsize\Large{40}{45}}
\renewcommand\LARGE{\@setfontsize\LARGE{45}{50}}
\renewcommand\huge{\@setfontsize\huge{50}{55}}
\renewcommand\Huge{\@setfontsize\Huge{55}{60}}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amsfonts}
\RequirePackage[absolute, overlay]{textpos}
\RequirePackage{mathtools}
\RequirePackage{booktabs}
\RequirePackage{xspace}
\RequirePackage[mode=buildnew]{standalone}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% graphicx options

\graphicspath{{figures/}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Got the dimensions from the website
% https://www.graphic-design-employment.com/a0-paper-dimensions.html
% A0 in 72dpi
\RequirePackage{geometry}
\geometry{paperwidth=2348px,paperheight=3370px}
\makeatletter
\setlength{\TP@visibletextwidth}{\textwidth-2\TP@innermargin}
\setlength{\TP@visibletextheight}{\textheight-2\TP@innermargin}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{xcolor}
% Paul Tol's  "Vibrant" color scheme
% https://personal.sron.nl/~pault/data/colourschemes.pdf
\definecolor{blue}{HTML}{0077BB}
\definecolor{cyan}{HTML}{33BBEE}
\definecolor{green}{HTML}{009988}
\definecolor{orange}{HTML}{EE7733}
\definecolor{red}{HTML}{CC3311}
\definecolor{magenta}{HTML}{EE3377}
\definecolor{grey}{HTML}{BBBBBB}
\definecolor{darkgrey}{HTML}{999999}

\definecolor{blockblue}{RGB}{38, 62, 66}
\definecolor{blockgrey}{RGB}{232, 232, 232}


\newcommand\black[1]{\textcolor{black}{#1}}
\newcommand\blue[1]{\textcolor{blue}{#1}}
\newcommand\cyan[1]{\textcolor{cyan}{#1}}
\newcommand\green[1]{\textcolor{green}{#1}}
\newcommand\orange[1]{\textcolor{orange}{#1}}
\newcommand\red[1]{\textcolor{red}{#1}}
\newcommand\magenta[1]{\textcolor{magenta}{#1}}
\newcommand\grey[1]{\textcolor{grey}{#1}}
\newcommand\darkgrey[1]{\textcolor{darkgrey}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\let\oscalebox\scalebox
\newcommand{\xscalebox}[2]{\oscalebox{#1}{\parbox{1.0\linewidth}{#2}}}
\let\scalebox\xscalebox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{tcolorbox}
\tcbuselibrary{skins}

\newtcolorbox{xbox}[2][]{
    colback=blockgrey,
    colframe=blockgrey,
    fonttitle=\bfseries,
    colbacktitle=blockgrey,
    enhanced, 
    coltitle=black,
    attach boxed title to top left={yshift=-3mm,xshift=0.5cm}, 
    sharpish corners,
    title={#2}, #1, 
    boxed title style={frame code={}}
}

\newtcolorbox{bluebox}[2][]{
    colback=blockgrey,
    colframe=blue,
    fonttitle=\bfseries,
    colbacktitle=blockgrey,
    enhanced, 
    coltitle=blue,
    attach boxed title to top left={yshift=-3mm,xshift=0.5cm}, 
    sharpish corners,
    title={#2}, #1, 
    boxed title style={frame code={}}
}
\newtcolorbox{redbox}[2][]{
    colback=blockgrey,
    colframe=red,
    boxrule=8pt,
    fonttitle=\bfseries,
    colbacktitle=blockgrey,
    enhanced, 
    coltitle=red,
    attach boxed title to top left={yshift=-3mm,xshift=0.5cm}, 
    sharpish corners,
    title={#2}, #1, 
    boxed title style={frame code={}}
}

\newtcolorbox{blackbox}[2][]{
    colback=blockgrey,
    colframe=black,
    fonttitle=\bfseries,
    colbacktitle=blockgrey,
    enhanced, 
    coltitle=black,
    attach boxed title to top left={yshift=-4mm,xshift=0.5cm}, 
    sharpish corners,
    title={#2}, #1, 
    boxed title style={frame code={}}
}

\newtcolorbox{blackfill}[1][]{
    before=\begin{center},
    after=\end{center},
    colback=black,
    colframe=white,
    coltext=white,
    fontupper=\bfseries\Large,
    sharpish corners,
    #1, 
    halign=center,
    valign=center,
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{tikz}
\usetikzlibrary{positioning, shapes, calc, arrows.meta}
\tikzset{
    cross/.pic = {
    \draw[rotate = 45] (-#1,0) -- (#1,0);
    \draw[rotate = 45] (0,-#1) -- (0, #1);
    }
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usetheme{Default} 
\tikzposterlatexaffectionproofoff

\defineblockstyle{blueblock}{
titleoffsety=10mm,
bodyverticalshift=10mm
}{
\draw[inner sep=\blockbodyinnersep, line width=0.5\blocklinewidth, draw=blockblue]
(blockbody.south west) rectangle (blocktitle.north east);
\draw[color=white, fill=blockblue, line width=0cm, draw=blockblue] (blocktitle.south west)
rectangle (blocktitle.north east);
}

\defineblockstyle{redblock}{
titleoffsety=10mm,
bodyverticalshift=10mm
}{
\draw[inner sep=\blockbodyinnersep, line width=0.5\blocklinewidth, draw=red]
(blockbody.south west) rectangle (blocktitle.north east);
\draw[color=white, fill=red, line width=0cm, draw=red] (blocktitle.south west)
rectangle (blocktitle.north east);
}

\defineblockstyle{greyblock}{
titleoffsety=10mm,
bodyverticalshift=10mm
}{
%\draw[inner sep=\blockbodyinnersep, line width=0.5\blocklinewidth, draw=grey] (blockbody.south west) rectangle (blocktitle.north east);
\draw[color=black, fill=blockgrey, line width=0cm, draw=blockgrey] (blocktitle.south west) rectangle (blocktitle.north east);
}

\defineblockstyle{nothingblock}{
titleoffsety=10mm,
bodyverticalshift=10mm
}{}

\useblockstyle{nothingblock}

\newcommand\myblock[3][MyBlock]{\useblockstyle{#1}\block{#2}{#3}\useblockstyle{nothingblock}}

\newcommand\mysubblock[2]{%
\begin{tikzpicture}%
\draw[line width=1mm, color=blue] (0,0) -- +(right:1\linewidth);
\node[anchor=west, color=blue, fill=white, inner xsep=1cm, font=\Large\bf] at (0,0) {#1};
\end{tikzpicture}%
}

\colorlet{backgroundcolor}{white}
\colorlet{blocktitlefgcolor}{black}
\colorlet{blocktitlebgcolor}{white}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Biblatex options

\RequirePackage[backend=biber, maxbibnames=10, maxcitenames=1, citestyle=authoryear-comp, bibstyle=authoryear, uniquelist=false, uniquename=false, dashed=false, natbib=true, sortcites=true, sorting=ynt]{biblatex}

\addbibresource{main.bib}
\renewcommand{\bibfont}{\normalfont\footnotesize}

% https://tex.stackexchange.com/questions/126226/how-do-i-instruct-fullcite-to-use-maxbibnames-rather-than-maxcitenames
\newcommand{\printpublication}[1]{\AtNextCite{\defcounter{maxnames}{99}}\fullcite{#1}}

\setlength{\bibhang}{0pt}
\setlength\bibitemsep{\baselineskip}

\renewbibmacro{in:}{}
\DeclareNameAlias{sortname}{given-family}

 % from \renewbibmacro*{date+extradate} in authoryear.bbx
 \renewbibmacro*{date+extradate}{}
  \renewbibmacro*{date}{%
    \iffieldundef{labelyear}
      {}
      {\printtext[parens]{%
         \iflabeldateisdate
           {\printfield{issue}%
            \setunit*{\addspace}%
            \printdateextra}
           {\printlabeldateextra}}}}

   % https://tex.stackexchange.com/questions/380299/change-position-of-year-field-in-authoryear-biblatex-style
  \renewbibmacro*{issue+date}{%
  \printtext[parens]{%
    \printfield{issue}%
    \setunit*{\addspace}%
     \printdate
     }%
  \newunit}

 % from \newbibmacro*{maintitle+booktitle} in standard.bbx
  \renewbibmacro*{maintitle+booktitle}{%
  \iffieldundef{booktitle}{}{\printfield{booktitle}}%
  \newunit}
  
  % from newbibmacro*{chapter+pages} in standard.bbx
  \renewbibmacro*{chapter+pages}{}
  % from newbibmacro*{note+pages} in standard.bbx  
  \renewbibmacro*{note+pages}{}

 % from \newbibmacro*{publisher+location+date} in standard.bbx
  \renewbibmacro*{publisher+location+date}{
  \printlist{publisher}
  \newunit
  \usebibmacro{date}
  \newunit}
  
  % from \newbibmacro*{journal+issuetitle} in standard.bbx
  \renewbibmacro*{journal+issuetitle}{%
  \usebibmacro{journal}%
  \newunit%
  \printfield{series}%
  \newunit
  \usebibmacro{volume+number+eid}%
  \newunit
  \usebibmacro{issue+date}%
  \newunit}

% https://github.com/plk/biblatex/issues/899
\DefineBibliographyStrings{english}{
  andothers = {\mkbibemph{et\addabbrvspace al\adddot}}
}

\DeclareFieldFormat*{title}{#1}
\DeclareFieldFormat*{booktitle}{\mkbibemph{#1}}
\DeclareListFormat*{publisher}{\mkbibemph{#1}}
\DeclareFieldFormat*{series}{\mkbibemph{#1}}

\DeclareCiteCommand{\citeauthor}{}
{
\ifdef{\theoldmaxnames}{}{\newcounter{oldmaxnames}}
\setcounter{oldmaxnames}{\themaxnames}
\setcounter{maxnames}{99}
\printnames{author}
\setcounter{maxnames}{\theoldmaxnames}
}{}{}

\DeclareCiteCommand{\citetitle}{}
{
\printfield{title}
}{}{}

\DeclareCiteCommand{\citedetails}{}
{\iffieldundef{booktitle}{
    \iffieldundef{journal}{
        \iffieldundef{note}{}{\printfield{note}}
    }{\printfield{journal}\iffieldundef{volume}{}{\printfield{volume}}}
}{\printfield{booktitle}}}{}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% defcommand

\makeatletter
\newcommand{\defcommand}[2]{%
  \ifdefined#1%
    \renewcommand{#1}{#2}%
  \else%
    \newcommand{#1}{#2}%
  \fi
}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Separators

\newcommand{\hsep}{
\begin{tikzpicture}
\draw[color=blockgrey, line width=0.3cm] (0,0) -- (\linewidth,0);
\end{tikzpicture}}
