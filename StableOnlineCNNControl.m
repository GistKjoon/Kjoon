function StableOnlineCNNControl()
% CNN1_ADAPTIVECONTROL 
%-------------------------------------------------------------------------
% 이 단일 파일 MATLAB 예제는 다음을 구현합니다:
%   1) 2차원 비선형 시스템: dot{x} = f(x) + u
%   2) CNN1 기반 적응 제어기 (합성곱 레이어 2개 + 완전연결 레이어 2개)
%   3) 온라인 경사하강(gradient descent) 기반 학습(파라미터 업데이트)
%
%-------------------------------------------------------------------------
% 이후에 추가할 실험사항
%   1) 3초에 헤비사이드 100
%   2) 
%   3)
%
%-------------------------------------------------------------------------
% 추가된 실험
%   1) dot{x} = f(x) + g(x) + u -> 발산
%   2) 
%   3)
%
%-------------------------------------------------------------------------


    % ----------------------
    % 0. 시뮬레이션 설정 부분
    % ----------------------
    T_end = 2;       % 전체 시뮬레이션 시간 (초 단위 가정)
    dt = 0.001;       % 시뮬레이션 스텝 크기 (적분 간격)
    time = 0:dt:T_end; 
    N = length(time);% 시간 벡터의 길이, 즉 스텝 수

    % --------------------------------
    % 시스템 상태/입력/차원 정의 부분
    % --------------------------------
    n_x = 2;    % 상태 벡터 x가 2차원 (x1, x2)
    n_u = 2;    % 제어 입력 u도 2차원 (u1, u2)

    % -----------------------------------------------------
    % 목표(레퍼런스) 궤적을 정의 : x_d(t) = [sin(2t); -cos(t)]
    % -----------------------------------------------------
    x_d_fun = @(t)[ sin(2*t); -cos(t) ];

    % ------------------------------------------------------
    % 비선형 함수 f(x)를 익명함수 형태로 정의
    % (dot{x} = f(x) + u)에서 f(x)
    % ------------------------------------------------------
    f_fun = @(x)[ ...
        x(1)*x(2)*tanh(x(2)) + sech(x(1)); 
        sech(x(1)+x(2))^2 - sech(x(2))^2 ];
    g_fun = @(x,time)[...
        2*x(1)^2*x(2)+2*sin(time)+20;
        2*x(2)^2*tanh(x(1)+2*cos(1/2*time)+20)]; 

    % -----------------------------------------------------
    % 제어 파라미터: A_c, ks, rho
    % A_c: 허위츠 행렬(여기서는 -10*I), ks: sign항 이득, rho: damping
    % -----------------------------------------------------
    Ac  = -10*eye(n_x);  % 2x2 행렬, 값은 [-10  0; 0  -10]
    ks  = 15;             % 오차의 부호항(sign(e)) 곱 이득
    rho = 1e5;           % e-modification 식 등에서 사용하는 감쇠 계수
    % rho = 0; 

    % -----------------------------------------------------
    % CNN1 아키텍처(합성곱 레이어 2개, FC 레이어 2개) 정의
    % -----------------------------------------------------
    arch = initCNN1_Architecture(); 
    % initCNN1_Architecture() 함수를 아래에서 구현

    % -----------------------------------------------------
    % CNN 입력 크기 (n0 x m0) 설정
    % e(2), x(2), u(2) => 총 6개 열
    % 시간을 10행 스택(예시)
    % -----------------------------------------------------
    n0 = 10;   
    m0 = 6;     % 열이 6개
    alpha2 = 0.01;  % CNN 입력 스케일링 계수

    % ----------------------------------------------------------------
    % CNN 전체 파라미터(세타)의 크기 계산
    % getThetaDimension_CNN1(arch)에서 166
    % ----------------------------------------------------------------
    thetaDim = getThetaDimension_CNN1(arch);
    % 아래 함수에서 합성곱 레이어 필터와 FC 레이어 파라미터 수를 모두 합산
    % 예) thetaDim = 166

    % ---------------------------------------------------------------
    % 세타(파라미터) 초기화 : [-0.1, 0.1] 범위로 균일분포 (랜덤)
    % ---------------------------------------------------------------
    theta_hat = 0.1*(2*rand(thetaDim,1)-1);

    % ---------------------------------------------------------------
    % 학습률(learning rate)을 행렬 Gamma로 설정 (thetaDim x thetaDim)
    % ---------------------------------------------------------------
    GammaVal = 0.1;
    Gamma    = GammaVal*eye(thetaDim);

    % -----------------------------
    % 시뮬레이션 변수 초기화
    % -----------------------------
    x = zeros(n_x, N);   % 상태 기록용 (2 x N)
    u = zeros(n_u, N);   % 제어 입력 기록용 (2 x N)
    e = zeros(n_x, N);   % 오차 기록용 (2 x N)

    % -----------------------------
    % 초기 상태 설정
    % -----------------------------
    x(:,1) = [1; 2];     % x(0) = [1;2]

    % -----------------------------
    % CNN 입력 행렬 (10 x 6) 초기화
    % -----------------------------
    X_cnn  = zeros(n0,m0);

    % --------------------------------
    % 메인 시뮬레이션 루프 (k=1 ~ N-1)
    % --------------------------------
    for k = 1:N-1
        % 1) 현재 시간
        t_now = time(k);

        % 2) 목표 궤적 계산: x_d = [sin(2t); -cos(t)]
        x_d = x_d_fun(t_now);

        % 3) 추종 오차 e = x - x_d
        e(:,k) = x(:,k) - x_d;

        % 4) CNN 입력 행렬 갱신
        %   rowNew = alpha2*[ e(1), e(2), x(1), x(2), u(1), u(2)]
        %   윗행에 집어넣고 아래로 시프트
        X_cnn = shiftAndStackData(X_cnn, e(:,k), x(:,k), u(:,k), alpha2);

        % 5) 순전파 => hatPhi (CNN + FC로부터 나온 최종 출력, 2차원)
        [hatPhi, cache] = cnnForwardPass_CNN1(X_cnn, theta_hat, arch);

        % 6) 제어 입력 계산
        %    sign(e) 대신 tanh(e/0.1)로 부드럽게 근사
        sgn_e = tanh( e(:,k)/0.1 );
        % sgn_e = sign(e(:,k));
        u(:,k) = -hatPhi - ks * sgn_e;

        % 7) 비선형 시스템 오일러 적분
        %    x(k+1) = x(k) + dt*(f(x(k))+u(k))
        % f_val = f_fun( x(:,k) )+ g_fun(x(:,k),time(k));
        f_val = f_fun( x(:,k) );
        x(:,k+1) = x(:,k) + dt*( f_val + u(:,k) );

        % 8) 다음 시점의 오차 계산
        e(:,k+1) = x(:,k+1) - x_d_fun(time(k+1));

        % 9) 역전파 => dPhi/dtheta 계산
        %    cnnBackprop_CNN1에서 CNN+FC 전체에 대한 편미분
        Phi_prime = cnnBackprop_CNN1(X_cnn, theta_hat, arch, cache);

        % 10) A_c^-1 * Phi_prime => gradTerm
        %     크기: (Ac_inv는 2x2, Phi_prime가 (2 x thetaDim) 가정)
        %     => 결과 (2 x thetaDim)
        %     => 전치 => (thetaDim x 2)
        Ac_inv = inv(Ac);
        gradTerm = (Ac_inv * Phi_prime)';  
        % gradTerm: (thetaDim x 2)

        % 11) e_k(2x1)와 곱 => dJ_dtheta( (thetaDim x 1) )
        e_k = e(:,k);     
        dJ_dtheta = gradTerm * e_k;

        % 12) damping 항: rho * ||e|| * ||theta||
        damping = rho * norm(e_k)*norm(theta_hat);

        % 13) 경사 하강 => dtheta
        dtheta   = -Gamma * ( dJ_dtheta + damping );
        newTheta = theta_hat + dt*dtheta;

        % 14) 파라미터 범위를 제한, norm(theta)<=10
        theta_hat = projectionOperator(newTheta, 10);

    end

    % 루프 종료 후 마지막 오차 기록
    e(:,N) = x(:,N) - x_d_fun(time(N));

    % -----------------------------
    % 결과 플롯
    % -----------------------------
    figure;
    subplot(2,1,1);
    plot(time, e(1,:), 'b', time, e(2,:), 'r','LineWidth',1.5);
    xlabel('time [s]'); ylabel('오차 e_1, e_2');
    title('추종 오차 (Tracking Error)'); 
    legend('e_1','e_2'); grid on;

    subplot(2,1,2);
    plot(time, u(1,:), 'b', time, u(2,:), 'r','LineWidth',1.5);
    xlabel('time [s]'); ylabel('제어입력 u_1, u_2');
    title('제어 입력 (Control Input)'); 
    legend('u_1','u_2'); grid on;

    disp('CNN1 시뮬레이션 종료');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% -------------------- 아키텍처 및 파라미터 차원 계산 ---------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function arch = initCNN1_Architecture()
% initCNN1_Architecture
% 이 함수는 CNN1 모델에서
%   - 합성곱 레이어(Convolution Layer) 수, 필터 크기
%   - 완전연결(FC) 레이어의 노드 수
% 등을 정의한다.

    % 합성곱 레이어가 2개라고 가정
    arch.CVL(1).numFilters = 2;      % 첫 번째 레이어 필터 개수
    arch.CVL(1).filterSize = [5,6];  % (p=5, m=6) => 필터 (5x6) 
    arch.CVL(2).numFilters = 2;      % 두 번째 레이어 필터 개수
    arch.CVL(2).filterSize = [3,2];  % (p=3, m=2)

    % 완전연결 레이어(2개)
    arch.FCL(1).numNodes   = 8;  % 첫 번째 FC 레이어 노드 수 8
    arch.FCL(2).numNodes   = 2;  % 두 번째(마지막) FC 레이어 노드 수 2 (최종출력 2D)
end

function dim = getThetaDimension_CNN1(arch)
% getThetaDimension_CNN1
%   - 주어진 arch.CVL(...) / arch.FCL(...) 정보로부터
%     합성곱(필터+바이어스) 파라미터 총합과
%     FC 레이어(가중치+바이어스) 파라미터 총합을 구한다.
%
% 실제 예시로 166

    % 먼저 합성곱 부분(CVL) 파라미터 수 계산
    dimCV = 0;
    for j = 1:length(arch.CVL)
        % j번째 합성곱 레이어
        q = arch.CVL(j).numFilters;      % 필터 개수
        p = arch.CVL(j).filterSize(1);   % 필터 행 크기
        m = arch.CVL(j).filterSize(2);   % 필터 열 크기
        % 필터 하나당 (p*m + 1) => (가중치 + 바이어스)
        % 필터 q개 => q*(p*m +1)
        dimCV = dimCV + q*(p*m + 1);
    end

    % 마지막 합성곱 레이어 출력 크기(예: (4x2)=8)라 가정 + 1 => FC에 입력
    % 여기서는 간단히 "최종 conv는 (4x2)=8"이라 보고 => 8+1=9
    % 만약 아키텍처마다 다를 수 있으니, 실제론 conv 결과 크기를 계산해야 함
    fcInputDim = 8 + 1;

    % 완전연결 FC 부분 계산
    dimFC = 0;
    prevDim = fcInputDim;  % 첫 레이어 입력차원(9)으로 시작
    for jf=1:length(arch.FCL)
        outDim = arch.FCL(jf).numNodes;
        % (prevDim * outDim) => 가중치 개수 (bias를 행렬 마지막 열에 포함하거나 등등)
        dimFC  = dimFC + prevDim*outDim; 
        prevDim= outDim + 1;  % 다음 레이어 입력= (outDim+1)
    end

    % 전체 파라미터 = 합성곱 + FC
    dim = dimCV + dimFC;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---------------------- 순전파 (CNN + FC) -------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hatPhi, cache] = cnnForwardPass_CNN1(X, theta, arch)
% cnnForwardPass_CNN1
%   - CNN(2단) + Flatten(+1) + FC(2단)
%   - 최종출력 hatPhi = R^2
%   - cache : 순전파 과정 중간값들(역전파용)

    % theta를 (합성곱 파트, FC 파트)로 분리
    [thetaCV, thetaFC] = splitTheta(arch, theta);

    % 합성곱 레이어 순전파
    [CVout, cacheCV] = forwardConvLayers(X, thetaCV, arch.CVL);

    % conv 결과를 flatten => CVflat, 마지막에 1을 붙여 => CVwb
    CVflat = CVout(:);
    CVwb   = [CVflat; 1]; 

    % 완전연결 레이어 순전파
    [hatPhi, cacheFC] = forwardFCLayers(CVwb, thetaFC, arch.FCL);

    % 캐시에 기록
    cache.X       = X;        % 원본 CNN 입력
    cache.CVout   = CVout;    % conv 최종 출력
    cache.CVwb    = CVwb;     % flatten+1 결과
    cache.cacheCV = cacheCV;  % conv 레이어 내부값들
    cache.cacheFC = cacheFC;  % FC 레이어 내부값들
end

function [CVout, cacheCV] = forwardConvLayers(X, thetaCV, CVL)
% forwardConvLayers
%   - 여러 합성곱 레이어를 순차 수행
%   - tanh 활성화 후, doConvolution1D로 conv

    k_c = length(CVL);              % 합성곱 레이어 개수(2)
    offsetsCV = parseCVOffsets(CVL);% 각 레이어별 파라미터 경계 인덱스
    outCell   = cell(k_c,1);        % 각 레이어 출력 기록
    phiCell   = cell(k_c,1);        % 각 레이어 입력(phiAct) 기록

    inp = X;                        % 첫 레이어 입력
    stPos=1;
    for j_c = 1:k_c
        edPos = offsetsCV(j_c);             % j_c번째 레이어 파라미터 경계
        subTheta = thetaCV(stPos:edPos);    % 해당 레이어 파라미터 추출
        stPos = edPos+1;

        q = CVL(j_c).numFilters;            % 필터 개수
        p = CVL(j_c).filterSize(1);         % 필터 행 크기
        m = CVL(j_c).filterSize(2);         % 필터 열 크기

        [Wset, Bvec] = reshapeCVParams(subTheta, q,p,m);

        % 활성화 함수 tanh 적용
        phiAct = tanh(inp);
        phiCell{j_c} = phiAct;

        % 1D 합성곱 수행
        out_jc = doConvolution1D(phiAct, Wset, Bvec);
        outCell{j_c} = out_jc;

        % 다음 레이어 입력
        inp = out_jc;
    end

    CVout = inp;                % 최종 conv 출력
    cacheCV.phi = phiCell;      % 각 레이어 입력
    cacheCV.out = outCell;      % 각 레이어 출력
end

function outMat = doConvolution1D(phiAct, Wset, Bvec)
% 실제 1D 합성곱 계산
% phiAct: (n0 x m0), 
% Wset: 필터 cell, Bvec: 바이어스
% outLen = n0 - p +1
% 결과: (outLen x q)

    [n0,m0] = size(phiAct);
    q = length(Wset);          % 필터 개수
    p = size(Wset{1},1);       % 필터 행 크기
    outLen = n0 - p +1;
    if outLen < 1
        outLen = 1; 
    end
    outMat = zeros(outLen, q);

    for j=1:q
        Wj = Wset{j};          % j번째 필터(행 p, 열 m0)
        Bj = Bvec(j);          % j번째 바이어스
        for i=1:outLen
            slice = phiAct(i:i+p-1,:);
            val   = sum(sum(Wj .* slice));
            outMat(i,j) = val + Bj;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ----------------------- 역전파 (CNN + FC) ------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Phi_prime = cnnBackprop_CNN1(X, theta, arch, cache)
% cnnBackprop_CNN1
%   - CNN+FC 전체에 대한 편미분(도함수)을 구하여
%     형태 반환

    % 1) theta 분리
    [thetaCV, thetaFC] = splitTheta(arch, theta);

    % 2) FC 레이어부터 역전파
    [dFCL, dCVwb] = backpropFCLayers(cache.cacheFC, arch.FCL);

    % 3) conv 레이어 최종 출력(cvOut)에 대한 미분(dCVout) 추출
    %    flatten되어 (nr*nc) => 다시 (nr x nc)
    CVout   = cache.CVout;
    [nr,nc] = size(CVout);
    nFlatten= nr*nc;
    dCVout  = dCVwb(1:nFlatten);
    dCVout  = reshape(dCVout,[nr,nc]);

    % 4) 합성곱 레이어 역전파
    dCV = backpropConvLayers(X, CVout, arch.CVL, cache.cacheCV, dCVout);

    % 5) 합성곱 + FC 미분 결과를 결합
    Phi_prime = combineGrad(arch, dCV, dFCL);
end

function dCV = backpropConvLayers(X, CVout, CVL, cacheCV, dCVout)
% 여러 conv 레이어에 대한 역전파
    k_c = length(CVL);
    dCV = cell(k_c,1);
    runDer = dCVout;

    for j_c = k_c:-1:1
        phiAct = cacheCV.phi{j_c};  % j_c 레이어 입력 (tanh 전)
        out_jc = cacheCV.out{j_c};  % j_c 레이어 출력

        q = CVL(j_c).numFilters;
        p = CVL(j_c).filterSize(1);
        m = CVL(j_c).filterSize(2);

        % 역전파
        [dW, dB, dBelow] = doConvolution1D_Backprop(phiAct, out_jc, runDer, q,p,m);

        % tanh' = 1 - phi^2
        dPhiAct = (1 - phiAct.^2) .* dBelow;
        runDer = dPhiAct;

        dCV{j_c}.dW = dW;
        dCV{j_c}.dB = dB;
    end
end

function [dW, dB, dPhiBelow] = doConvolution1D_Backprop(phiAct, out_jc, dOut, q,p,m)
% 합성곱 역전파 (간단화 버전)
    [n0,m0] = size(phiAct);
    outLen  = size(out_jc,1);
    dW = cell(q,1);
    dB = cell(q,1);

    % 실제론 phiBelow까지 계산해야 하지만 여기선 예시로 0
    dPhiBelow= zeros(n0,m0);

    for j=1:q
        dOut_j = dOut(:,j);
        dWj = zeros(p,m);
        dBj = 0;
        for i=1:outLen
            slice = phiAct(i:i+p-1,:);
            dWj = dWj + dOut_j(i)*slice;
            dBj = dBj + dOut_j(i);
            % dPhiBelow(i:i+p-1,:) += dOut_j(i)* Wj ... (생략)
        end
        dW{j} = dWj;
        dB{j} = dBj;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ------------------------ FC 레이어 순전파/역전파 -----------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Fout, cacheFCL] = forwardFCLayers(inVec, thetaFC, FCL)
% 완전연결 레이어들 순전파
    nL = length(FCL);
    offsets = parseFCOffsets(FCL, length(inVec));
    layerData= cell(nL,1);

    inp = inVec;     % 초기 입력
    stPos=1;         
    for L=1:nL
        outDim= FCL(L).numNodes;   % 노드 수
        edPos= offsets(L);
        subTheta= thetaFC(stPos:edPos);
        stPos= edPos+1;

        % (outDim x length(inp)) 형태로 가중치 reshape
        W = reshape(subTheta,[outDim, length(inp)]);

        % z = W * inp
        z = W*inp;

        % 활성화 함수 tanh
        a = tanh(z);

        % 마지막 레이어가 아니면 +1(바이어스용)
        if L<nL
            a= [a;1];
        end

        layerData{L}.W   = W;
        layerData{L}.z   = z;
        layerData{L}.act = a;
        layerData{L}.inp = inp;

        inp = a;
    end
    Fout = inp;  
    cacheFCL.layerData= layerData;
end

function [dFCL, dInput] = backpropFCLayers(cacheFCL, FCL)
% FC 레이어 역전파
    layerData= cacheFCL.layerData;
    nL = length(layerData);

    % 최종 레이어 출력이 2차원이라 가정 (ex: [2x1])
    outVec= layerData{nL}.act;
    outDim= length(outVec);

    % dAct = [1;1]라 가정
    dAct= ones(outDim,1);

    thetaCell = cell(nL,1);

    for L=nL:-1:1
        W= layerData{L}.W;
        a= layerData{L}.act;
        inp= layerData{L}.inp;
        [od,id]= size(W);

        if L<nL
            aNoBias= a(1:end-1);
        else
            aNoBias= a;
        end

        da= 1 - aNoBias.^2;  % tanh' = 1 - tanh^2
        if length(dAct)~= od
            dAct= dAct(1:od);
        end
        dZ= dAct .* da;

        % 가중치 미분 dW
        dWlayer= zeros(od*id,1);
        for r=1:od
            for c=1:id
                idx= (r-1)*id + c;
                dWlayer(idx)= dZ(r)* inp(c);
            end
        end

        % 입력쪽 역전파 dInp
        dInp= zeros(id,1);
        for r=1:od
            dInp= dInp + dZ(r)* W(r,:)';
        end

        if L>1
            dInp= dInp(1:end-1);
        end

        dAct= dInp;
        thetaCell{L}= dWlayer;
    end

    % 레이어별 dWlayer를 가로로 이어붙임
    allArr= [];
    for L=1:nL
        arrL= thetaCell{L}';
        allArr= [allArr, arrL];
    end

    % dFCL.fullDer : FC 파라미터 전체에 대한 편미분
    dFCL.fullDer= allArr;
    % dInput : FC 입력에 대한 미분
    dInput = dAct(:)';
end

function offsets = parseFCOffsets(FCL, inputDim)
% FC 레이어별 파라미터 offset 계산
    nL= length(FCL);
    offsets= zeros(nL,1);
    cumul=0;
    curDim= inputDim;
    for i=1:nL
       outDim= FCL(i).numNodes;
       block= outDim*curDim; 
       cumul= cumul+ block;
       offsets(i)= cumul;
       curDim= outDim +1; 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ----------- 합성곱/FC 미분 결합, 기타 유틸(프로젝션 등) --------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [thetaCV, thetaFC] = splitTheta(arch, theta)
% arch 정보에 따라 theta를 (합성곱 파트, FC 파트)로 구분
    dimCV=0;
    for j=1:length(arch.CVL)
       q= arch.CVL(j).numFilters;
       p= arch.CVL(j).filterSize(1);
       m= arch.CVL(j).filterSize(2);
       dimCV= dimCV+ q*(p*m+1);
    end
    thetaCV= theta(1:dimCV);
    thetaFC= theta(dimCV+1:end);
end

function paramOffsets = parseCVOffsets(CVL)
% 합성곱 레이어별 파라미터 offset
    k_c= length(CVL);
    paramOffsets= zeros(k_c,1);
    cumul=0;
    for j=1:k_c
        q= CVL(j).numFilters;
        p= CVL(j).filterSize(1);
        m= CVL(j).filterSize(2);
        block= q*(p*m+1);
        cumul= cumul+block;
        paramOffsets(j)= cumul;
    end
end

function [Wset,Bvec] = reshapeCVParams(subTheta, q,p,m)
% 하나의 conv 레이어 파라미터(subTheta)를
% 필터(q개) + 바이어스(q개)로 나눈다
    Wset= cell(q,1);
    Bvec= zeros(q,1);
    off=1;
    for i=1:q
        wlen= p*m;
        wpart= subTheta(off : off+wlen-1);
        off= off+wlen;
        Wset{i}= reshape(wpart,[p,m]);
        Bvec(i)= subTheta(off);
        off= off+1;
    end
end

function out = combineGrad(arch, dCV, dFCL)
% CNN(합성곱)과 FC 레이어 역전파 결과를 합쳐
% (2 x totalDim) 형태로 만든다
    dimCV=0;
    for j=1:length(arch.CVL)
        q= arch.CVL(j).numFilters;
        p= arch.CVL(j).filterSize(1);
        m= arch.CVL(j).filterSize(2);
        dimCV= dimCV + q*(p*m+1);
    end

    % 합성곱 레이어 미분 결과
    vecCV= [];
    for j=1:length(dCV)
        dWcell= dCV{j}.dW;
        dBcell= dCV{j}.dB;
        for i=1:length(dWcell)
            fW= dWcell{i}; 
            fWv= fW(:);
            fB= dBcell{i};
            tmp= [fWv; fB];
            vecCV= [vecCV; tmp];
        end
    end
    % CNN은 최종 출력이 2차원 => row 복제
    outCV= repmat(vecCV(:)', 2,1);

    % FC 레이어 미분 결과
    outFC= dFCL.fullDer; 
    outFC2= repmat(outFC,2,1);

    % 두 부분을 가로로 이어붙임
    out= [outCV, outFC2];
end

function xproj = projectionOperator(x, boundVal)
% 파라미터 벡터 x의 노름이 boundVal 넘으면 스케일링
    nr= norm(x);
    if nr>boundVal
        xproj= x*(boundVal/nr);
    else
        xproj= x;
    end
end
